export CUDA_VISIBLE_DEVICES=1

lrs=(2.5e-5)
multi_levels=("note")

for multi_level in "${multi_levels[@]}"
do
for lr in "${lrs[@]}"
do
for fold in {0..3}
do
    python virtuoso/__main__.py \
    -yml ymls/shared/label19/han_bigger256_concat.yml \
    -data m2pf_allround/composition4fold_perffeatonly/${fold} \
    --label_file_path label_2round_mean_reg_19_with0_rm_highstd0.json \
    --batch_size 8 \
    -dev cuda \
    --num_labels 19 \
    --checkpoints_dir /root/v2/muzic/virtuosonet/checkpoints_2rounds_align/composition4fold_perffeatonly/level/${multi_level}/fold${fold}/ \
    --lr $lr \
    --no_augment  \
    --multi_level $multi_level

done
done
done
