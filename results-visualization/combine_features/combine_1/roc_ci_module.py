import pandas as pd  
import numpy as np 
from sklearn.metrics import roc_curve 
import matplotlib.pyplot as plt  
from sklearn import metrics

def roc_ci(y_test, y_score, clr, fillclr, column, select):
    fpr1, tpr1, _ = roc_curve(y_test, y_score)  
    
    # auc
    roc_auc1 = metrics.auc(fpr1, tpr1)  
    
    # 95% CI 
    n_bootstraps = 1000  
    rng = np.random.RandomState(42)  
    tprs = []  
    aucs = []  
    mean_fpr = np.linspace(0, 1, 100)  
    
    for i in range(n_bootstraps):  
        sample = rng.choice(len(y_test), len(y_test), replace=True)  
        fpr, tpr, _ = roc_curve(y_test[sample], y_score[sample]) 
        
        roc_auc = metrics.auc(fpr, tpr)  
        interp_tpr = np.interp(mean_fpr, fpr, tpr)  
        interp_tpr[0] = 0.0  
        tprs.append(interp_tpr)  
        aucs.append(roc_auc) 

    # calculate tprs 95% CI
    tprs = np.array(tprs)  
    mean_tpr = tprs.mean(axis=0)  
    std_tpr = tprs.std(axis=0)  
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)   
    tprs_lower = np.maximum(mean_tpr - 1.96 *std_tpr, 0)
    
    # plot ROC and CI
    plt.figure(figsize=(50/25.4, 50/25.4))  
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 7  
    plt.plot(fpr1, tpr1, color=clr, lw=1, label=f'Combined {column}\n(AUC={roc_auc1:.3f})') 
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=fillclr, alpha=0.3)   
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')   
    plt.title(f'ROC: {column} AUC>{select}')  
    plt.legend(loc="lower right")  
    #save
    plt.savefig(f'all_results/combine_rocs/{column}_{select}.jpg', format='jpg',dpi=600) 
