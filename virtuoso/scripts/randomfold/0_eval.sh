export CUDA_VISIBLE_DEVICES=6


lrs=(2.5e-05)
# for lr in "${lrs[@]}"
# do
#     checkpoints=("yml_path_ymls/shared/label19/han_bigger256_concat.yml_no_augment_True_label_file_path_label_2round_mean_reg_19_with0_rm_highstd0.json_batch_size_8_lr_2.5e-05_intermediate_loss_False")
#     for checkpoint in "${checkpoints[@]}"
#     do  
#         echo $checkpoint
#             checkpointt="/root/v2/muzic/virtuosonet/checkpoints_2rounds_align/randomfold/base/foldx/${checkpoint}"
#             python virtuoso/eval_main.py \
#             -yml ymls/shared/label19/han_bigger256_concat.yml \
#             --label_file_path label_2round_mean_reg_19_with0_rm_highstd0.json \
#             -data m2pf_allround/random8fold/foldx \
#             --batch_size 1 \
#             --checkpoint $checkpointt \
#             --num_labels 19 \
#             --n_folds 8 \
#             -reTrain "true" \
#             --device "cuda" \
#             --make_log "False" | tee checkpoints_2rounds_align/randomfold/base/han_concat_bigger256_noaugment_lr${lr}_eval.log
#     done
# done


# for lr in "${lrs[@]}"
# do
#     checkpoints=("yml_path_ymls/shared/label19/han_bigger256_concat.yml_multi_level_note_no_augment_True_label_file_path_label_2round_mean_reg_19_with0_rm_highstd0.json_batch_size_8_lr_2.5e-05_intermediate_loss_False")
#     for checkpoint in "${checkpoints[@]}"
#     do  
#         echo $checkpoint
#             checkpointt="/root/v2/muzic/virtuosonet/checkpoints_2rounds_align/randomfold/note/foldx/${checkpoint}"
#             python virtuoso/eval_main.py \
#             -yml ymls/shared/label19/han_bigger256_concat.yml \
#             --label_file_path label_2round_mean_reg_19_with0_rm_highstd0.json \
#             -data m2pf_allround/random8fold/foldx \
#             --batch_size 1 \
#             --checkpoint $checkpointt \
#             --num_labels 19 \
#             --n_folds 8 \
#             -reTrain "true" \
#             --device "cuda" \
#             --multi_level "note" \
#             --make_log "False" | tee checkpoints_2rounds_align/randomfold/note/han_concat_bigger256_noaugment_lr${lr}_eval.log
#     done
# done

for lr in "${lrs[@]}"
do
    checkpoints=("yml_path_ymls/shared/label19/han_bigger256_concat.yml_no_augment_True_label_file_path_label_2round_mean_reg_19_with0_rm_highstd0.json_batch_size_8_lr_2.5e-05_intermediate_loss_False")
    for checkpoint in "${checkpoints[@]}"
    do  
        echo $checkpoint
            checkpointt="/root/v2/muzic/virtuosonet/checkpoints_2rounds_align/performerfold/foldx/${checkpoint}"
            python virtuoso/eval_main.py \
            -yml ymls/shared/label19/han_bigger256_concat.yml \
            --label_file_path label_2round_mean_reg_19_with0_rm_highstd0.json \
            -data m2pf_allround/performer4fold/foldx \
            --batch_size 1 \
            --checkpoint $checkpointt \
            --num_labels 19 \
            --n_folds 4 \
            -reTrain "true" \
            --device "cuda" \
            --make_log "False" | tee checkpoints_2rounds_align/performerfold/han_concat_bigger256_noaugment_lr${lr}_eval.log
    done
done