# Rena Ahn
# calculate_stats.py (Python, Anaconda3 Interpreter)
# 6 November 2024
# Calculates summary statistics using imputed data

import os
import numpy as np
import pandas as pd

# Constant Variables
FOLDER_PATH = 'impute/data_imputed_year'
PATIENT_FOLDER_PATH = FOLDER_PATH + '/patients'
NEW_FOLDER_PATH = 'summary_stats/year'


### Main Code ###
stat_dfs = {}
func_list = ['count', 'min', 'median', 'max', 'mean', 'sum', 'std', 'skew', 'mode']   #mode currently not calculated

# calculate statistics from imputed files
for filename in os.listdir(PATIENT_FOLDER_PATH):
    file_path = PATIENT_FOLDER_PATH + '/' + filename
    patient_df = pd.read_csv(file_path, na_values=['', '-'])

    name = filename.strip('.csv')
    data_df = pd.DataFrame()   #dataframe with pefr data
    func_dict = {}
    for col in patient_df.columns:
        if col != 'date':
            data_df[col] = patient_df[col]
            func_dict[col] = func_list[0:8]

    stat_df = data_df.agg(func_dict)   #statistics dataframe
    stat_dfs[name] = stat_df

# statistic DataFrames -> .csv files
for name, df in stat_dfs.items():
    new_file_path = NEW_FOLDER_PATH + '/' + name + '_stats.csv'
    df.to_csv(new_file_path)

print('Number of Files: ', len(stat_dfs))