import shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_

from MidiBERT.finetune_model import SequenceClassification, SequenceClassificationHierarchical, SequenceClassificationAlign


class FinetuneTrainer:
    def __init__(self, midibert, train_dataloader, valid_dataloader, test_dataloader, layer, 
                lr, class_num, hs, testset_shape, cpu, cuda_devices=None, model=None, addons = False, 
                virtuosonet=None, head_type = 'attentionhead', output_type = 'note', align=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        print('   device:',self.device)
        self.midibert = midibert
        self.addons = addons
        self.align = align
        self.layer = layer
        self.SeqClass = True

        if model != None:    # load model
            print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model, addons?', addons)
            if addons:
                self.virtuosonet = virtuosonet
                self.model = SequenceClassificationHierarchical(self.midibert, self.virtuosonet, class_num, 
                                                                hs, head_type=head_type, output_type=output_type).to(self.device)
            elif align:
                self.virtuosonet = virtuosonet
                self.model = SequenceClassificationAlign(self.midibert, self.virtuosonet, class_num, hs, head_type=head_type).to(self.device)
            else:
                self.model = SequenceClassification(self.midibert, class_num, hs, head_type=head_type).to(self.device)


        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader
        
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_func = nn.MSELoss()

        self.testset_shape = testset_shape

        self.mask_start_bar = -1
        self.mask_end_bar = -1
    
    def compute_loss(self, predict, target, loss_mask, seq):
        loss = self.loss_func(predict, target.float())
        return loss
 
    def train(self):
        self.model.train()
        train_loss, train_r2 = self.iteration(self.train_data, 0, self.SeqClass)
        return train_loss, train_r2

    def valid(self):
        self.model.eval()
        valid_loss, valid_r2 = self.iteration(self.valid_data, 1, self.SeqClass)
        return valid_loss, valid_r2

    def test(self):
        self.model.eval()
        test_loss, test_r2, all_output = self.iteration(self.test_data, 2, self.SeqClass)
        return test_loss, test_r2, all_output

    def iteration(self, training_data, mode, seq):

        total_yhat = []
        total_y = []
        total_r2, total_cnt, total_loss = 0, 0, 0

        if mode == 2: # testing
            cnt = 0
        for num_processed, x in tqdm.tqdm(enumerate(training_data), total = len(training_data)):  # (batch, 512, 768)
            if self.addons:
                if len(x) == 6:
                    x, y, addons, note_location, data_len, found_addon_idxs = x
                else:
                    x, y, addons, note_location, data_len = x
                    found_addon_idxs = None
                batch = x.shape[0]
                x, y, addons = x.to(self.device), y.to(self.device), addons.to(self.device)
            elif self.align:
                x, y, addons, data_len = x
                batch = x.shape[0]
                x, y, addons = x.to(self.device), y.to(self.device), addons.to(self.device)
            else:
                x, y = x
                batch = x.shape[0]
                x, y = x.to(self.device), y.to(self.device)     # seq: (batch, 512, 4), (batch) / token: , (batch, 512)

            # avoid attend to pad word
            if not seq:
                attn = (y != 0).float().to(self.device)   # (batch,512)
            else:   
                attn = torch.ones((batch, 512)).to(self.device)     # attend each of them
            
            if self.addons and self.mask_start_bar > -1:
                bar_location = note_location['measure']
                for batch_idx, bar_locs in enumerate(bar_location):
                    for bar_loc in bar_locs[:data_len[batch_idx]]:
                        if self.mask_start_bar <= bar_loc < self.mask_end_bar:
                            x[batch_idx, bar_loc] = torch.tensor(self.midibert.mask_word_np).to(self.device)

            if self.addons:
                y_hat = self.model(x, attn, self.layer, addons, note_location, data_len, found_addon_idxs)
            elif self.align:
                y_hat = self.model(x, attn, self.layer, addons, data_len)
            else:
                y_hat = self.model(x, attn, self.layer)     # seq: (batch, class_num) / token: (batch, 512, class_num)
            # apply sigmoid
            y_hat = torch.sigmoid(y_hat)
            if mode == 2:
                cnt += batch

            # calculate losses
            loss = self.compute_loss(y_hat, y, attn, seq)
            total_loss += loss.item()

            # udpate only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

            # r2 
            total_cnt += y.shape[0]
            total_yhat.append(y_hat.cpu().detach().numpy())
            total_y.append(y.cpu().detach().numpy())
        
        total_yhat = np.concatenate(total_yhat, axis=0)
        total_y = np.concatenate(total_y, axis=0)
        total_r2 = r2_score(total_y, total_yhat)

        if mode == 2:
            return round(total_loss/len(training_data),4), round(total_r2,4), (total_yhat, total_y)
        return round(total_loss/len(training_data),4), round(total_r2,4)


    def save_checkpoint(self, epoch, train_r2, valid_r2, 
                        valid_loss, train_loss, is_best, filename):
        state_dict = self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict()
        state = {
            'epoch': epoch + 1,
            'state_dict': state_dict, 
            'valid_r2': valid_r2,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'train_r2': train_r2,
            'optimizer' : self.optim.state_dict()
        }
        torch.save(state, filename)

        best_mdl = filename.split('.')[0]+'_best.ckpt'
        
        if is_best:
            shutil.copyfile(filename, best_mdl)

