import pandas as pd  
import numpy as np 

from combine3_module import operate_auc
from table_module import table

def main(): 
    exp_index = 'data/mask_group_differential_Haoran/jianwai/jianwai_MSD.csv'
    con_index = 'data/mask_group_differential_Haoran/zhengwei/zhengwei_MSD.csv'

    results_dir = 'data/mean_median_results'
    mask_jianwai_dir = 'data/mask_group_differential_Haoran/jianwai' 
    mask_zhengwei_dir = 'data/mask_group_differential_Haoran/zhengwei'

    all_lst=[]
    for i in ['original_firstorder_Mean', 'original_firstorder_Median']:
        for m in [0.7,0.75,0.8]:
            all_lst.append(operate_auc(i,m, 
                                   exp_index, con_index, 
                                   results_dir, mask_jianwai_dir, mask_zhengwei_dir))

    lst=[]
    aucs = ['0.7','0.75','0.8']
    for j in range(3):
        lst.append(table(all_lst[j], 'predicted_prob', 'Mean', f'AUC>{aucs[j]}'))
        lst.append(table(all_lst[j+3], 'predicted_prob', 'Median', f'AUC>{aucs[j]}'))

    sa = pd.DataFrame(lst)
    sa.to_csv('all_results/auc_table_results.csv', index=False)


if __name__ == "__main__":
    main()