import os
from glob import glob
from rocs_module import roc

def main():
    # source address
    results_dir = 'data/stat_results'
    mask_jianwai_dir = 'data/mask_group_differential_Haoran/jianwai'
    mask_zhengwei_dir = 'data/mask_group_differential_Haoran/zhengwei'
    # output address
    final_res_dir = 'all_results/ROCs_seperated'
    # parameters 
    width = 45
    height_ratio = 1

    # loop over all datas
    for res in os.listdir(results_dir):
        if res.endswith('.xlsx'):  #
            name = os.path.splitext(res)[0]

            # experimental group
            pattern_jianwai = f'*{name}*'
            matches_jianwai = glob(os.path.join(mask_jianwai_dir, pattern_jianwai))[0]

            # control group
            pattern_zhengwei = f'*{name}*'
            matches_zhengwei = glob(os.path.join(mask_zhengwei_dir, pattern_zhengwei))[0]

            roc(matches_jianwai, matches_zhengwei, name, width, height_ratio, final_res_dir)

if __name__ == "__main__":
    main()