export PYTHONPATH="."

export CUDA_VISIBLE_DEVICES=1

OUTPUT_TYPES=(total_note_cat)
NET_PARAMS_PATHS=(./Data/virtuoso_params/layer1lstm.json)
BATCH_SIZES=(8)


for batch_size in ${BATCH_SIZES[@]}; do
for output_type in ${OUTPUT_TYPES[@]}; do
for net_params_path in ${NET_PARAMS_PATHS[@]}; do    
    python MidiBERT/finetune_cv.py --task=percepiano --name=composition4_addonhierarchy --start_cv_idx=0 --end_cv_idx=4 \
     --net_params_path=${net_params_path} --addons_path=./Data/CP_data/percepiano_composition4split_addon --head_type=attentionhead --output_type=${output_type} \
        --batch_size=${batch_size}
done
done
done