import pandas as pd  
import numpy as np 

from combine2_module import operate
from roc_ci2_module import roc_ci

def main(): 
    exp_index = 'data/mask_group_differential_Haoran/jianwai/jianwai_MSD.csv'
    con_index = 'data/mask_group_differential_Haoran/zhengwei/zhengwei_MSD.csv'

    results_dir = 'data/mean_median_results'
    mask_jianwai_dir = 'data/mask_group_differential_Haoran/jianwai' 
    mask_zhengwei_dir = 'data/mask_group_differential_Haoran/zhengwei'

    all_lst=[]
    for i in ['original_firstorder_Mean', 'original_firstorder_Median']:
        for m in ['forward','backward','bidirection']:
            all_lst.append(operate(i,m, 
                                   exp_index, con_index, 
                                   results_dir, mask_jianwai_dir, mask_zhengwei_dir))

    l1 = ['Mean', 'Median']
    l2 = ['forward','backward','bidirection']
    for i in range(3):
        y1=all_lst[i]
        y2=all_lst[i+3]
        roc_ci(y1['group'], y1['predicted_prob'], None, None, l1[0], l2[i])
        roc_ci(y2['group'], y2['predicted_prob'], 'darkorange', 'moccasin', l1[1], l2[i])


if __name__ == "__main__":
    main()