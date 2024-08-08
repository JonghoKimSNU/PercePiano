import json
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_std_score(predictions, true_values, label_stds, acceptable_range=0.05):
    """
    Evaluate multi-label regression predictions.
    Args:
    - predictions (numpy array): Predicted values for each label.
    - true_values (numpy array): True values for each label.
    - acceptable_range (float): Acceptable range as a percentage (0.05 for 5%).
    Returns:
    - within_range_pct (list of floats): Percentage of predictions within acceptable range for each label.
    """
    num_labels = predictions.shape[1] - 1  # Get the number of labels
    within_range_pct = []

    for label in range(num_labels):
        label_preds = predictions[:, label]
        label_true = true_values[:, label]
        label_std = label_stds[:, label]

        # Calculate percentage of predictions within acceptable range
        lower_bound = label_true - acceptable_range * label_std
        upper_bound = label_true + acceptable_range * label_std
        within_range = np.logical_and(label_preds >= lower_bound, label_preds <= upper_bound)
        within_range_pct.append(np.mean(within_range) * 100.0)

    return within_range_pct

# caculate average value for each label (19 labels) in train set, than evaluate on the test set
def evaluate_random(labels, label_stds, test_list):
    # fork keys in labels, in test_list -> test set, else: train set.
    train_list = [f for f in labels.keys() if f not in test_list]
    print("train_list", len(train_list), "test_list", len(test_list))
    # get the average value for each label in train set
    train_labels = np.array([labels[f] for f in train_list])
    label_mean = np.mean(train_labels, axis=0) # shape: (19,)
    label_mean = np.tile(label_mean, (len(test_list), 1)) # shape: (len(test_list), 19)
    test_labels = np.array([labels[f] for f in test_list])
    test_label_stds = np.array([label_stds[f] for f in test_list])
    # print shape
    # print("train_labels", train_labels.shape)
    # print("label_mean", label_mean.shape)
    # print("test_labels", test_labels.shape)
    # print("test_label_stds", test_label_stds.shape)
    r2 = r2_score(test_labels, label_mean)
    mse = mean_squared_error(test_labels, label_mean)
    std_score_1 = evaluate_std_score(label_mean, test_labels, test_label_stds, 1)
    std_score_05 = evaluate_std_score(label_mean, test_labels, test_label_stds, 0.5)
    std_score_01 = evaluate_std_score(label_mean, test_labels, test_label_stds, 0.1)
    std_score_1 = np.mean(std_score_1)
    std_score_05 = np.mean(std_score_05)
    std_score_01 = np.mean(std_score_01)
    print("r2,mse,std_score_1,std_score_05,std_score_01")
    print(r2, mse, std_score_1, std_score_05, std_score_01)
    return r2, mse, std_score_1, std_score_05, std_score_01

if __name__ == "__main__":
    

    label_files = json.load(open("/root/v2/muzic/virtuosonet/label_2round_mean_reg_19_with0_rm_highstd0.json"))
    label_std_files = json.load(open("/root/v2/muzic/virtuosonet/label_2round_std_reg_19_with0_rm_highstd0.json"))

    random_test_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/random8fold/0/test")
    random_test_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in random_test_list if ".mid" in f]
    # performer4fold_test_list = open(f"/root/v2/muzic/music-xai/processed/xaiperformersplit_data_2round_mean_reg_19/0/test.id").read().splitlines()
    performer4fold_test_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/performer4fold/0/test")
    performer4fold_test_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in performer4fold_test_list if ".mid" in f]
    evaluate_random(label_files, label_std_files, performer4fold_test_list)
    evaluate_random(label_files, label_std_files, random_test_list)
    composition_test_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/composition4fold/0/test")
    composition_test_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in composition_test_list if ".mid" in f]
    evaluate_random(label_files, label_std_files, composition_test_list)
