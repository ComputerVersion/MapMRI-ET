import pandas as pd  
import numpy as np 
import os  
from glob import glob
from sklearn.linear_model import LogisticRegression

from stepwise3_module import stepwise

def combine_features(ex_dir, con_dir, res_dir, name, col, df1, df2):
    # load data
    experimental_data = pd.read_csv(ex_dir)  
    control_data = pd.read_csv(con_dir) 

    df1[name+'_'+col] = experimental_data[col]
    df2[name+'_'+col] = control_data[col]

def operate_step(col, method, 
            exp_index = 'data/mask_group_differential_Haoran/jianwai/jianwai_MSD.csv', 
            con_index = 'data/mask_group_differential_Haoran/zhengwei/zhengwei_MSD.csv', 
            results_dir = 'data/mean_median_results',
            mask_jianwai_dir = 'data/mask_group_differential_Haoran/jianwai', 
            mask_zhengwei_dir = 'data/mask_group_differential_Haoran/zhengwei'):
    
    exp = pd.read_csv(exp_index)  
    con = pd.read_csv(con_index) 
    df1 = pd.DataFrame({  
        'sample': exp['Unnamed: 0'],
    }) 
    df2 = pd.DataFrame({  
        'sample': con['Unnamed: 0'],
    })
    
    # loop over all files  
    for res in os.listdir(results_dir):  
        if res.endswith('.csv'):  # csv 
            name = os.path.splitext(res)[0] 
            
            # experimental group 
            pattern_jianwai = f'*{name}*'  
            ex_dir = glob(os.path.join(mask_jianwai_dir, pattern_jianwai))[0]  
            
            # control group
            pattern_zhengwei = f'*{name}*'  
            con_dir = glob(os.path.join(mask_zhengwei_dir, pattern_zhengwei))[0] 

            res_dir = 'data/mean_median_results/'+res
            combine_features(ex_dir, con_dir, res_dir, name, col, df1, df2)

    # contact to a DataFrame，jianwai=0, zhengwei=1
    all_data = pd.concat([df1.assign(group=0),  
                        df2.assign(group=1)],  
                        ignore_index=True) 

    all_data['predicted_prob'] = 0 
    
    # seperate  
    X = all_data.drop(columns=['sample', 'group','predicted_prob']) 
    y = all_data['group']  
    
    fe = stepwise(X, y, method)
    X=X[fe]

    # init 
    model = LogisticRegression()  
    
    # train
    model.fit(X, y) 

    predicted_probabilities = model.predict_proba(X)[:, 1]  
    all_data['predicted_prob'] = predicted_probabilities

    return all_data

def auc_features(ex_dir, con_dir, res_dir, name, select, col, df1, df2):
    # load data  
    experimental_data = pd.read_csv(ex_dir)  
    control_data = pd.read_csv(con_dir) 
    results=pd.read_csv(res_dir)

    # select 'AUC' > ** 'property' column
    filtered_properties = results.loc[results['AUC'] > select, 'property'].tolist()

    for prop in filtered_properties:
        if prop == col:
            df1[name+'_'+prop] = experimental_data[prop]
            df2[name+'_'+prop] = control_data[prop]

def operate_auc(col, auc, 
            exp_index = 'data/mask_group_differential_Haoran/jianwai/jianwai_MSD.csv', 
            con_index = 'data/mask_group_differential_Haoran/zhengwei/zhengwei_MSD.csv', 
            results_dir = 'data/mean_median_results',
            mask_jianwai_dir = 'data/mask_group_differential_Haoran/jianwai', 
            mask_zhengwei_dir = 'data/mask_group_differential_Haoran/zhengwei'):
    
    exp = pd.read_csv(exp_index)  
    con = pd.read_csv(con_index) 
    df1 = pd.DataFrame({  
        'sample': exp['Unnamed: 0'],
    }) 
    df2 = pd.DataFrame({  
        'sample': con['Unnamed: 0'],
    })
    
    # loop over all files  
    for res in os.listdir(results_dir):  
        if res.endswith('.csv'):  # csv 
            name = os.path.splitext(res)[0] 
            
            # experimental group 
            pattern_jianwai = f'*{name}*'  
            ex_dir = glob(os.path.join(mask_jianwai_dir, pattern_jianwai))[0]  
            
            # control group
            pattern_zhengwei = f'*{name}*'  
            con_dir = glob(os.path.join(mask_zhengwei_dir, pattern_zhengwei))[0] 

            res_dir = 'data/mean_median_results/'+res
            auc_features(ex_dir, con_dir, res_dir, name, auc, col, df1, df2)

    # contact to a DataFrame，jianwai=0, zhengwei=1
    all_data = pd.concat([df1.assign(group=0),  
                        df2.assign(group=1)],  
                        ignore_index=True) 

    all_data['predicted_prob'] = 0 
    
    # seperate  
    X = all_data.drop(columns=['sample', 'group','predicted_prob']) 
    y = all_data['group']  

    # init 
    model = LogisticRegression()  
    
    # train
    model.fit(X, y) 

    predicted_probabilities = model.predict_proba(X)[:, 1]  
    all_data['predicted_prob'] = predicted_probabilities

    return all_data