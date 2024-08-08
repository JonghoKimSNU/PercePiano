import json
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os

# {"/root/v2/muzic/virtuosonet/data/all_2rounds/Beethoven_WoO80_var23_8bars_2_13.mid": [[0.5951685309410095, 0.6461150050163269, 0.4260059595108032, 0.7015647292137146, 0.542615532875061, 0.594172477722168, 0.6798538565635681, 0.566872775554657, 0.4175911545753479, 0.3785551190376282, 0.5343195796012878, 0.5926269888877869, 0.6625360250473022, 0.6859924793243408, 0.6619433164596558, 0.5650195479393005, 0.5054871439933777, 0.6643012166023254, 0.6362239122390747]]

LABEL_LIST19 = [
    "Timing_Stable_Unstable",
    "Articulation_Long_Short",
    "Articulation_Soft_cushioned_Hard_solid",
    "Pedal_Sparse/dry_Saturated/wet",
    "Pedal_Clean_Blurred",
    "Timbre_Even_Colorful",
    "Timbre_Shallow_Rich",
    "Timbre_Bright_Dark",
    "Timbre_Soft_Loud",
    "Dynamic_Sophisticated/mellow_Raw/crude",
    "Dynamic_Little_dynamic_range_Large_dynamic_range",
    "Music_Making_Fast_paced_Slow_paced",
    "Music_Making_Flat_Spacious",
    "Music_Making_Disproportioned_Balanced",
    "Music_Making_Pure_Dramatic/expressive",
    "Emotion_&_Mood_Optimistic/pleasant_Dark",
    "Emotion_&_Mood_Low_Energy_High_Energy",
    "Emotion_&_Mood_Honest_Imaginative",
    "Interpretation_Unsatisfactory/doubtful_Convincing"
]

pred1_file = "checkpoints_2rounds_align/performerfold/level/note/fold0/yml_path_ymls/shared/label19/han_bigger256_concat.yml_multi_level_measure_no_augment_True_label_file_path_label_2round_mean_reg_19_with0_rm_highstd0.json_batch_size_8_lr_2.5e-05_intermediate_loss_False_2402121736/results_performfold.json"
pred2_file = "checkpoints_2rounds_align/performerfold_perffeatonly_wopedal/level/note/fold0/yml_path_ymls/shared/label19/han_bigger256_concat.yml_multi_level_note_no_augment_True_label_file_path_label_2round_mean_reg_19_with0_rm_highstd0.json_batch_size_8_lr_2.5e-05_intermediate_loss_False_2408021704/results_performfold.json"
label_file = "label_2round_mean_reg_19_with0_rm_highstd0.json"
label = json.load(open(label_file))
pred1 = json.load(open(pred1_file))
pred2 = json.load(open(pred2_file))

assert len(pred1) == len(pred2)
names = []
labels = [] # (data size, 19 labels)
pred1s = [] # (data size, 19 labels)
pred2s = [] # (data size, 19 labels)
# calculate r2
for k, v in pred1.items():
    name = os.path.basename(k).rstrip(".mid")
    names.append(name)
    pred1s.append(v)
    pred2s.append(pred2[k])
    labels.append(label[name][:19])

pred1s = np.array(pred1s).squeeze(1)
pred2s = np.array(pred2s).squeeze(1)
labels = np.array(labels)
print(pred1s.shape, pred2s.shape, labels.shape)

# Calculate MSE scores
mse_scores1 = np.mean((labels - pred1s) ** 2, axis=1)
mse_scores2 = np.mean((labels - pred2s) ** 2, axis=1)

# Identify performances where pred1 is better than pred2
better_performances = [i for i in range(len(mse_scores1)) if mse_scores1[i] < mse_scores2[i]]

# Sort the performances and get the top 20%
top_performances = sorted(better_performances, key=lambda x: mse_scores1[x])[:int(len(better_performances) * 0.2)]

print(len(top_performances), len(better_performances), len(mse_scores1))
# Prepare results
results = {
    names[i]: {
        "ours": pred1s[i].tolist(),
        "baseline": pred2s[i].tolist(),
        "label": labels[i].tolist(),
        "better_labels": [(LABEL_LIST19[j], ",".join(["ours", f"{pred1s[i][j]:.4f}", "baseline", f"{pred2s[i][j]:.4f}", "label", f"{labels[i][j]:.4f}"]))
                  for j in range(19) if mse_scores1[i] + 0.004 < mse_scores2[i]]
    }
    for i in top_performances
}
print(results)
# count the number of better labels
count = 0
for k, v in results.items():
    better_labels = v["better_labels"]
    better_labels_count = len(better_labels)
    count += better_labels_count
print(count)
    
# Save results to JSON
with open("case_study/comparison_results_performer4fold.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to comparison_results.json")

