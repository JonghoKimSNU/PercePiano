export PYTHONPATH="."

export CUDA_VISIBLE_DEVICES=2

OUTPUT_TYPES=(measure)
NET_PARAMS_PATHS=(./Data/virtuoso_params/layer1lstm.json)
BATCH_SIZES=(8)

python MidiBERT/finetune_cv.py --task=percepiano --name=composition4_default \
        --start_cv_idx=0 --end_cv_idx=1 \
        --batch_size=8 --head_type=attentionhead
