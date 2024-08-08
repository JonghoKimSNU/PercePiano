# bert base
from scipy.stats import ttest_rel
import numpy as np

def read_file(file, feature_level = False):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    metric_dict = {"std_score_1": [], "std_score_05": [], "std_score_01": [], "r2": [], "mse": []}
    # 19 labels. each label has 5 metrics
    whole_metric_dict = [{"std_score_1": [], "std_score_05": [], "std_score_01": [], "r2": [], "mse": []} for _ in range(19)]
    # fine the lines next to the line "std_score_1, std_score_05, std_score_01, r2, mse"
    for i in range(len(lines)):
        if lines[i-1] == "final scores":
            continue
        if lines[i] == "std_score_1, std_score_05, std_score_01, r2, mse":
            metrics = lines[i+1]
            metrics = metrics.split()
            metric_dict["std_score_1"].append(float(metrics[0]))
            metric_dict["std_score_05"].append(float(metrics[1]))
            metric_dict["std_score_01"].append(float(metrics[2]))
            metric_dict["r2"].append(float(metrics[3]))
            metric_dict["mse"].append(float(metrics[4]))
            # from lines[i+2] to lines[i+2+19]
            for j in range(19):
                metric = lines[i+2+j]
                metric = metric.split()
                whole_metric_dict[j]["std_score_1"].append(float(metric[0]))
                whole_metric_dict[j]["std_score_05"].append(float(metric[1]))
                whole_metric_dict[j]["std_score_01"].append(float(metric[2]))
                whole_metric_dict[j]["r2"].append(float(metric[3]))
                whole_metric_dict[j]["mse"].append(float(metric[4]))
    return metric_dict, whole_metric_dict


def ttest(file1, file2, feature_level = False):
    metric_dict1, whole_metric_dict1 = read_file(file1, feature_level = False)
    metric_dict2, whole_metric_dict2 = read_file(file2, feature_level = False)
    print(file1, file2)
    for key in metric_dict1:
        t, p = ttest_rel(metric_dict1[key], metric_dict2[key])
        print(f"{key}: t={t}, p={p}")
        # print(metric_dict1[key])
        # print(metric_dict2[key])
        print(f"{key}: mean1={np.mean(metric_dict1[key])}, mean2={np.mean(metric_dict2[key])}")
        print()
    # with whole metric dict, print ttest result for each label. shape: 19 * 5
    each_ttest_results = []
    for i in range(19):
        for key in whole_metric_dict1[i]:
            ttest_results = []    
            t, p = ttest_rel(whole_metric_dict1[i][key], whole_metric_dict2[i][key])
            ttest_results.append(p)
        each_ttest_results.append(ttest_results)
    # print with shape 19 * 5 matrix
    each_ttest_results = np.array(each_ttest_results)
    # print(each_ttest_results)

if __name__ == "__main__":

    """performer4fold"""
    print("performerfold")
    baseline = "checkpoints_2rounds_align/performerfold_perffeatonly_wopedal/level/note/han_concat_bigger256_noaugment_lr2.5e-05_note_eval.log"
    baseline_sa = "checkpoints_2rounds_align/performerfold/level/note/han_concat_bigger256_noaugment_lr2.5e-05_note_eval.log"
    baseline_sa_han = "checkpoints_2rounds_align/performerfold/level/note/han_concat_bigger256_noaugment_lr2.5e-05_measure_eval.log"
    ttest(baseline, baseline_sa)
    ttest(baseline_sa, baseline_sa_han)
    ttest(baseline, baseline_sa_han)  # all significant
    """composition4fold"""
    print("composition4fold")
    baseline = "checkpoints_2rounds_align/composition4fold_perffeatonly/level/note/han_concat_bigger256_noaugment_lr2.5e-05_note_eval.log"
    baseline_sa = "checkpoints_2rounds_align/composition4fold/level/note/han_concat_bigger256_noaugment_lr2.5e-05_note_eval.log"
    baseline_sa_han = "checkpoints_2rounds_align/composition4fold/level/measure/han_concat_bigger256_noaugment_lr2.5e-05_measure_eval.log"
    ttest(baseline, baseline_sa_han) 
    ttest(baseline, baseline_sa)  
    ttest(baseline_sa, baseline_sa_han) 

