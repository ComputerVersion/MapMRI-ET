import pandas as pd  
import numpy as np 

from combine3_module import operate_step
from table_module import table

def main(): 
    exp_index = 'data/mask_group_differential_Haoran/jianwai/jianwai_MSD.csv'
    con_index = 'data/mask_group_differential_Haoran/zhengwei/zhengwei_MSD.csv'

    results_dir = 'data/mean_median_results'
    mask_jianwai_dir = 'data/mask_group_differential_Haoran/jianwai' 
    mask_zhengwei_dir = 'data/mask_group_differential_Haoran/zhengwei'

    all_lst=[]
    for i in ['original_firstorder_Mean', 'original_firstorder_Median']:
        for m in ['forward','backward','bidirection']:
            all_lst.append(operate_step(i,m, 
                                   exp_index, con_index, 
                                   results_dir, mask_jianwai_dir, mask_zhengwei_dir))

    lst=[]
    methods = ['forward','backward','bidirection']
    for j in range(3):
        lst.append(table(all_lst[j], 'predicted_prob', 'Mean', methods[j]+' stepwise'))
        lst.append(table(all_lst[j+3], 'predicted_prob', 'Median', methods[j]+' stepwise'))

    sa = pd.DataFrame(lst)
    sa.to_csv('all_results/stepwise_table_results.csv', index=False)


if __name__ == "__main__":
    main()