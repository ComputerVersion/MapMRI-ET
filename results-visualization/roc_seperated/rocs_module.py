import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc(file1, file2, name, width=45, height_ratio=1, res_dir='ROCs_seperated'):
    # load data
    experimental_data = pd.read_csv(file1)
    control_data = pd.read_csv(file2)

    # contact data to a DataFram
    all_data = pd.concat([experimental_data.assign(group=0),
                          control_data.assign(group=1)],
                         ignore_index=True)

    filtered_properties = ['original_firstorder_Mean', 'original_firstorder_Median']

    all_labels = all_data['group']

    # draw roc curves 
    plt.figure(figsize=(width/25.4, height_ratio*width/25.4))  
    plt.rcParams['font.family'] = 'Times New Roman' # set font 
    plt.rcParams['font.size'] = 6  # font size

    for prop in filtered_properties:
        all_scores=all_data[prop]
        # calculate roc curves
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)  
        roc_auc = auc(fpr, tpr) 
        if roc_auc < 0.5:
            fpr, tpr, thresholds = roc_curve(1-all_labels, all_scores) 
            roc_auc = auc(fpr, tpr) 
        plt.plot(fpr, tpr, lw=1, label=prop.rsplit('_', 1)[1]+' (AUC=%0.3f)' % roc_auc)  
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')   
    plt.title('ROCs of '+name)  
    plt.legend(loc="lower right")  

    # save
    plt.savefig(f'{res_dir}/roc_{name}.jpg', format='jpg',dpi=600)