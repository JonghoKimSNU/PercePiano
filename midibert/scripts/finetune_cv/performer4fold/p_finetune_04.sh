export PYTHONPATH="."

export CUDA_VISIBLE_DEVICES=7

python MidiBERT/finetune_cv.py --task=percepiano --name=performer_default \
        --start_cv_idx=0 --end_cv_idx=4 \
        --batch_size=8 --head_type=attentionhead

