# Rena Ahn
# combine_stats.py (Python, Anaconda3 Interpreter)
# 6 November 2024
# Calculates statistics using imputed data and combines with demographic table

import os
import numpy as np
import pandas as pd

# Constant Variables
FOLDER_PATH = 'impute/data_imputed_all'
PATIENT_FOLDER_PATH = FOLDER_PATH + '/patients'
NEW_FOLDER_PATH = 'summary_stats/combine'

### Main Code ###
stat_dfs = {}
all_columns = ['ID', 'count', 'min', 'median', 'max', 'mean', 'sum', 'std', 'skew']   #include 'mode' later
func_list = all_columns[1:]

# calculate statistics from imputed files
for filename in os.listdir(PATIENT_FOLDER_PATH):
    file_path = PATIENT_FOLDER_PATH + '/' + filename
    patient_df = pd.read_csv(file_path, na_values=['', '-'])

    name = filename.strip('.csv')
    data = list([])
    data_df = pd.DataFrame(columns=['pefr'])

    for col in patient_df.columns:
        if col != 'date':
            new_data = pd.DataFrame()
            new_data['pefr'] = patient_df[col].dropna()
            if data_df.shape[0] == 0:
                data_df = new_data
            else:
                data_df = pd.concat([data_df, new_data], axis=0, ignore_index=True)
    stat_df = data_df.agg(func_list[:8])   #statistics dataframe
    stat_dfs[name] = stat_df

# build a collective dataframe with the statistics of all patients
summary = {}   #dictionary containing all statistic summary values
for col in all_columns:   #initializing keys
    summary[col] = []

for name, df in stat_dfs.items():   #populating values
    summary['ID'].append(name)
    
    for i in range(df.shape[0]):
        summary[func_list[i]].append(df.iloc[i, 0])
    
all_stats = pd.DataFrame()   #dataframe with all statistics
for col, values in summary.items():
    all_stats[col] = values

# Processing demographic files -> DataFrames
demographic_df = pd.read_csv(FOLDER_PATH + '/demographic_all.csv')
demographic_df = demographic_df.merge(all_stats, on='ID', how='inner')

demographic_df.to_csv(NEW_FOLDER_PATH + '/demographic_all.csv', index=False)

demographic_df_part = demographic_df.drop(['BCODE', 'UID'], axis=1)
demographic_df_part.to_csv(NEW_FOLDER_PATH + '/demographic.csv', index=False)

print()
print('*-- Collective Demographic and Statistic Data --*')
print(demographic_df)
