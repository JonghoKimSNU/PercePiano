export PYTHONPATH="."

export CUDA_VISIBLE_DEVICES=2


NET_PARAMS_PATHS=(./Data/virtuoso_params/layer1lstm.json)
OUTPUT_TYPES=(note measure total_note_cat)
BATCH_SIZES=(8)

for output_type in ${OUTPUT_TYPES[@]}; do
for net_params_path in ${NET_PARAMS_PATHS[@]}; do    
    for batch_size in ${BATCH_SIZES[@]}; do
    python MidiBERT/finetune_cv.py --task=percepiano --name=performer_addonhierarchy --start_cv_idx=0 --end_cv_idx=4 \
     --net_params_path=${net_params_path} --addons_path=./Data/CP_data/percepiano_performersplit_addon --batch_size=${batch_size} --head_type=attentionhead \
     --output_type=${output_type} \
     --ckpt result/finetune/percepiano_performer_addonhierarchy/layer1lstm_lr1e-05_bs${batch_size}_attentionhead_${output_type}/xx/model_best.ckpt \
     --do_eval
done
done
done

for batch_size in "${batch_sizes[@]}"; do
    python MidiBERT/finetune_cv.py --task=percepiano --name=performer_woaddon --start_cv_idx=0 --end_cv_idx=4 \
    --batch_size=${batch_size} \
     --ckpt result/finetune/percepiano_performer_default/lr1e-05_bs${batch_size}_attentionhead/xx/model_best.ckpt \
     --do_eval \
      --head_type=attentionhead
done