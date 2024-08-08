export CUDA_VISIBLE_DEVICES=1

lrs=(2.5e-05)
# LEVELS=("note")

# for lr in "${lrs[@]}"
# do
# for level in "${LEVELS[@]}"
# do
#     checkpoints=("yml_path_ymls/shared/label19/han_bigger256_concat.yml_multi_level_${level}_no_augment_True_label_file_path_label_2round_mean_reg_19_with0_rm_highstd0.json_batch_size_8_lr_2.5e-05_intermediate_loss_False")
#     for checkpoint in "${checkpoints[@]}"
#     do  
#         echo $checkpoint        
#             checkpointt="/root/v2/muzic/virtuosonet/checkpoints_2rounds_align/performerfold_perffeatonly_wopedal/level/${level}/foldx/${checkpoint}"
#             python virtuoso/eval_main.py \
#             -yml ymls/shared/label19/han_bigger256_concat.yml \
#             --label_file_path label_2round_mean_reg_19_with0_rm_highstd0.json \
#             -data m2pf_allround/performer4fold_perffeatonly_wopedal/foldx \
#             --batch_size 1 \
#             --checkpoint $checkpointt \
#             --num_labels 19 \
#             --n_folds 1 \
#             -reTrain "true" \
#             --device "cuda" \
#             --multi_level ${level} \
#             --make_log "False" \
#             --return_each
#             # | tee checkpoints_2rounds_align/performerfold_perffeatonly_wopedal/level/${level}/han_concat_bigger256_noaugment_lr${lr}_${level}_eval.log
#     done
# done
# done

LEVELS=("measure")

for lr in "${lrs[@]}"
do
for level in "${LEVELS[@]}"
do
    checkpoints=("yml_path_ymls/shared/label19/han_bigger256_concat.yml_multi_level_${level}_no_augment_True_label_file_path_label_2round_mean_reg_19_with0_rm_highstd0.json_batch_size_8_lr_2.5e-05_intermediate_loss_False")
    for checkpoint in "${checkpoints[@]}"
    do  
        echo $checkpoint        
            checkpointt="/root/v2/muzic/virtuosonet/checkpoints_2rounds_align/performerfold/level/note/foldx/${checkpoint}"
            python virtuoso/eval_main.py \
            -yml ymls/shared/label19/han_bigger256_concat.yml \
            --label_file_path label_2round_mean_reg_19_with0_rm_highstd0.json \
            -data m2pf_allround/performer4fold/foldx \
            --batch_size 1 \
            --checkpoint $checkpointt \
            --num_labels 19 \
            --n_folds 1 \
            -reTrain "true" \
            --device "cuda" \
            --multi_level ${level} \
            --make_log "False" \
            --return_each
    done
done
done