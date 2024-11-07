# Rena Ahn
# read_files.py (Python, Anaconda3 Interpreter)
# 6 November 2024
# Reads patient files (.xlsx) and converts them to .csv files

import os
import numpy as np
import pandas as pd

# Constant Variables
MIN_NUM_VALUES = 1
PATIENT_FOLDER_PATH = 'data_org/SCH_asthma_114'
NEW_FOLDER_PATH = 'data_read_all'
NEW_PATIENT_FOLDER_PATH = NEW_FOLDER_PATH + '/patients'

# 114_medinfo file -> DataFrame
demographic_df = pd.DataFrame(pd.read_excel('data_org/Demographic.xlsx'))

# new UID values
new_uid = np.where(demographic_df['UID2'].isna(), demographic_df['UID1'], demographic_df['UID2'])
demographic_df['UID'] = pd.Series(new_uid)
demographic_df = demographic_df.drop(['UID1', 'UID2'], axis=1)

# listed patient id
patient_ids = list(demographic_df['ID'])

#print(demographic_df.info())

# individual patient files -> DataFrame
patient_dfs = {}

for filename in os.listdir(PATIENT_FOLDER_PATH):
    # converting patient file -> Dataframe
    file_path = PATIENT_FOLDER_PATH + '/' + filename
    patient_df = pd.DataFrame(pd.read_excel(file_path))

    patient_id = patient_df.iloc[2, 0] if patient_df.iloc[1, 0][0] == 'A' \
                                       else patient_df.iloc[1, 0]
    if patient_id in patient_ids:
        # new dataframe 
        new_df = pd.DataFrame()
        new_df['date'] = patient_df.iloc[2:, 1]

        # add columns if missing values are less than the minimum threshold
        num_values = patient_df.shape[0] - 2
        
        if patient_id == 'SB-102':     #data for file SB-102 located in different columns
            if num_values - patient_df.iloc[2:, 9].isna().sum() > MIN_NUM_VALUES:
                new_df['pefr_am'] = patient_df.iloc[2:, 9]
            if num_values - patient_df.iloc[2:, 10].isna().sum() > MIN_NUM_VALUES:
                new_df['pefr_pm'] = patient_df.iloc[2:, 10]
            if num_values - patient_df.iloc[2:, 11].isna().sum() > MIN_NUM_VALUES:
                new_df['pefr_other'] = patient_df.iloc[2:, 11]
        else:
            if num_values - patient_df.iloc[2:, 6].isna().sum() > MIN_NUM_VALUES:
                new_df['pefr_am'] = patient_df.iloc[2:, 6]
            if num_values - patient_df.iloc[2:, 7].isna().sum() > MIN_NUM_VALUES:
                new_df['pefr_pm'] = patient_df.iloc[2:, 7]
            if num_values - patient_df.iloc[2:, 8].isna().sum() > MIN_NUM_VALUES:
                new_df['pefr_other'] = patient_df.iloc[2:, 8]

        if new_df.shape[1] > 1:   #ensures there is PEFR data
            patient_dfs[patient_id] = new_df

# checking patients removed and ensuring counts match
new_patient_ids = list(patient_dfs.keys())
patients_removed = len(patient_ids) - len(new_patient_ids)

valid_rows = demographic_df['ID'].isin(new_patient_ids)
removed_df = demographic_df[~valid_rows]

# updating patient id list and demographic DataFrame
patient_ids = new_patient_ids
demographic_df = demographic_df[valid_rows]

# storing dataframes -> csv files to data_read folder
demographic_df.to_csv(NEW_FOLDER_PATH + '/demographic_all.csv', index=False)

demographic_df_part = demographic_df.drop(['BCODE', 'UID'], axis=1)
demographic_df_part.to_csv(NEW_FOLDER_PATH + '/demographic.csv', index=False)

for name, df in patient_dfs.items():
    new_file_path = NEW_PATIENT_FOLDER_PATH + '/' + name + '.csv'
    df.to_csv(new_file_path, index=False)

# Output Summary
print('Number of Patients Removed: ', patients_removed)
print('Number of Patients Found with Removed Ids: ', removed_df.shape[0])
