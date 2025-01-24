import pandas as pd  
import numpy as np 
from sklearn.metrics import roc_curve, auc  

# calculate
def calculate_metrics(y_true, y_pred, pos_label=1):  
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))  
    tn = np.sum((y_true == 0) & (y_pred == 0))  
    fp = np.sum((y_true == 0) & (y_pred == pos_label))  
    fn = np.sum((y_true == pos_label) & (y_pred == 0))  
      
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  
      
    return sensitivity, specificity, ppv, npv  
  
# select best threshold by Youden index 
def select_best_threshold_by_youden(df, score_col, label_col, pos_label=1):  
    thresholds = np.unique(df[score_col])  
    youden_scores = []  
      
    for threshold in thresholds:  
        y_pred = (df[score_col] >= threshold).astype(int)  
        sensitivity, specificity, _, _ = calculate_metrics(df[label_col], y_pred, pos_label)  
        youden = sensitivity + specificity - 1  
        youden_scores.append((threshold, youden))  
      
    best_threshold, best_youden = max(youden_scores, key=lambda x: x[1])  
    return best_threshold  
  
# use Bootstrap to get 95% CI
def bootstrap_ci(data, metric_func, n_bootstraps=1000, ci=0.95):  
    metric_values = []  
    for _ in range(n_bootstraps):  
        sample_idx = np.random.choice(data.index, size=len(data), replace=1)  
        sample = data.loc[sample_idx]  
        y_true = sample['label']  
        y_pred = sample['pred_label']  
        metric_value = metric_func(y_true, y_pred)  
        metric_values.append(metric_value)  
      
    lower_bound = np.percentile(metric_values, (1 - ci) / 2 * 100)  
    upper_bound = np.percentile(metric_values, (1 + ci) / 2 * 100)  
    return lower_bound, upper_bound  

###########################################################################

def table(df, score, property, method):
    df['label']=df['group']
    df=df[[score,'label']]
    
    # select best threshold 
    best_threshold = select_best_threshold_by_youden(df, score, 'label')  
    
    # prediction label 
    df.loc[:, 'pred_label'] = (df[score] >= best_threshold).astype(int)
    
    # calculate
    sensitivity, specificity, ppv, npv = calculate_metrics(df['label'], df['pred_label'])   
    
    # 95% CI 
    sen_ci = bootstrap_ci(df, lambda y_true, y_pred: calculate_metrics(y_true, y_pred)[0])  
    spec_ci = bootstrap_ci(df, lambda y_true, y_pred: calculate_metrics(y_true, y_pred)[1])  
    npv_ci = bootstrap_ci(df, lambda y_true, y_pred: calculate_metrics(y_true, y_pred)[3])  
    ppv_ci = bootstrap_ci(df, lambda y_true, y_pred: calculate_metrics(y_true, y_pred)[2])  

    #auc 
    def cal_auc(label_, score_):
        fpr, tpr, thresholds = roc_curve(label_, score_)  
        return auc(fpr, tpr) 
    roc_auc = cal_auc(df['label'], df[score])

    auc_ci = bootstrap_ci(df, lambda y_true, y_pred: cal_auc(y_true, y_pred))

    # make DataFrame  
    re = {  
        'selected property': property,
        'selected method': method,
        'property': score,  
        'AUC': roc_auc,
        '95% CI of AUC': f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
        'best_threshold': best_threshold,
        'sensitivity': sensitivity,
        '95% CI of sensitivity': f"[{sen_ci[0]:.4f}, {sen_ci[1]:.4f}]",
        'specificity': specificity,
        '95% CI of specificity': f"[{spec_ci[0]:.4f}, {spec_ci[1]:.4f}]",
        'npv': npv,
        '95% CI of npv': f"[{npv_ci[0]:.4f}, {npv_ci[1]:.4f}]",
        'ppv': ppv,
        '95% CI of ppv': f"[{ppv_ci[0]:.4f}, {ppv_ci[1]:.4f}]"
    }   
    return re