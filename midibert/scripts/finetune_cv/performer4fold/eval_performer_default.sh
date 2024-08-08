export PYTHONPATH="."

export CUDA_VISIBLE_DEVICES=3
batch_sizes=(8)

for batch_size in "${batch_sizes[@]}"; do
    python MidiBERT/finetune_cv.py --task=percepiano --name=performer_woaddon --start_cv_idx=0 --end_cv_idx=4 \
    --batch_size=${batch_size} \
     --ckpt result/finetune/percepiano_performer_default/lr1e-05_bs${batch_size}_attentionhead/xx/model_best.ckpt \
     --do_eval \
      --head_type=attentionhead
done