export CUDA_VISIBLE_DEVICES=6

# lrs=(5e-5 1e-4)
lrs=(2.5e-5)

for lr in "${lrs[@]}"
do
for fold in {0..7}
do
    python virtuoso/__main__.py \
    -yml ymls/shared/label19/han_bigger256_concat.yml \
    -data m2pf_allround/random8fold/${fold} \
    --label_file_path label_2round_mean_reg_19_with0_rm_highstd0.json \
    --batch_size 8 \
    -dev cuda \
    --num_labels 19 \
    --checkpoints_dir /root/v2/muzic/virtuosonet/checkpoints_2rounds_align/randomfold/base/fold${fold}/ \
    --lr $lr \
    --no_augment  \

done
done