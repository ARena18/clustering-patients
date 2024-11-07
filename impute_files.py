# Rena Ahn
# impute_files.py (Python, Anaconda3 Interpreter)
# 6 November 2024
# Imputes missing data in patient data

"""
Due to the nature of the data (PEFR values can be affected by time), our team
decided to limit imputation.
Imputation will only occur on data with gaps of < 14 days missing.
As a result, we assume the impact of our imputations are negligible.
Accordingly, there is limited testing of imputation techniques.

We chose to interpolate linearly, forward fill, then backward fill.
The mean is not robust, and due to the large scale (and at times range) of the
data, we felt the median may not capture the trend at the current time period.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constant Variables
MIN_NUM_VALUES = 1
FOLDER_PATH = 'data_read_all'
PATIENT_FOLDER_PATH = FOLDER_PATH + '/patients'
NEW_FOLDER_PATH = 'impute/data_imputed_all'
NEW_PATIENT_FOLDER_PATH = NEW_FOLDER_PATH + '/patients'

"""
Function to check the size of gaps
Returns true if the patient dataframe has a gap of more than 14 missing data
values in a row
Otherwise, returns false
Additionally returns the dictionary containing the indices of the large gaps
Pre : patient is a pandas DataFrame object
"""
def checkLargeGaps(patient):
    num_rows, num_cols = patient.shape
    all_counts = {}
    large_gaps = {}
    large = False

    for i in range(1, num_cols):
        all_counts[i] = []
        c = 0
        large_gaps[i] = []
        for j in range(num_rows):
            if patient.iloc[j, i] > 0:   #checks data is not missing
                if c > 14:
                    all_counts[i].append(c)               
                    large_gaps[i].append((j - c, j))
                    large = True
                c = 0
            else:
                c += 1
    
        if c > 14:
            all_counts[i].append(c)
            large_gaps[i].append((num_rows-c-1, num_rows-1))
            large = True
    """
    if not impute:
        print(all_counts)
        print(large_gaps)
        print()
    
    if impute:
        print(all_counts)
        print(large_gaps)
    """
    return large, large_gaps

"""
Function to choose and impute the largest chunk of data (without a gap of more
than 14 days in between)
Pre : patient is a pandas DataFrame object,
      gaps is a dictionary containing tupes of the start and end indices of each large gap
"""
def imputeLargestChunk(patient, gaps):
    col_names = list(patient.columns)
    join_dfs = []

    # iterate through each pefr data column
    for i in range(1, patient.shape[1]):
        max = 0   #size of largest chunk
        max_start = 0   #start index of gap after largest chunk
        max_end = 0   #end index of gap before largest chunk
          # start (after) and end (before) index of largest gap
        prev_end = 0   #end index of gap before current chunk

        # find largest chunk of current column
        for start, end in gaps[i]:
            size = start - prev_end
            if size > max:
                max = size
                max_start = start
                max_end = prev_end
            prev_end = end

        if len(gaps[i]) == 0:   #when there are no large gaps
            max = len(patient[col_names[i]]) - patient[col_names[i]].isna().sum()
            max_start = max
            max_end = 0

        if max > MIN_NUM_VALUES: #checks size of chunk
            # make dataframe of the chunk
            join_df = pd.DataFrame()
            #print(max_end, " ", max_start)
            join_df[col_names[0]] = patient[col_names[0]].iloc[max_end:max_start]
            join_df[col_names[i]] = patient[col_names[i]].iloc[max_end:max_start].interpolate(method='linear').ffill().bfill()
            join_dfs.append(join_df)
    
    if len(join_dfs) == 0:   #checks there is data
        return None
    
    # outer join all dataframes of individual chunks
    df_joined = join_dfs[0]
    
    #print(len(join_dfs))
    for i in range(1, len(join_dfs)):
        df_joined = pd.merge(df_joined, join_dfs[i], on=col_names[0], how='outer')

    check, nouse = checkLargeGaps(df_joined)
    if not check:
        df_joined.ffill().bfill()

    return df_joined

"""
Function to interpolate linearly
Returns the interpolated dataframe
Pre : patient is a pandas DataFrame object
"""
def imputeLinear(patient):
    impute_cols = list(patient.columns)[1:] #columns to impute data (float data type)
    
    df_linear = patient
    df_linear[impute_cols] = patient[impute_cols].interpolate(method='linear')
    df_linear = df_linear.ffill().bfill() #fills remaining missing data

    #print(df_linear.isna().sum())
    return df_linear


### Main Code ###
patient_dfs = {}
small_gap_dfs = []
large_gap_dfs = []
num_files = 0

# Processing patient files -> DataFrames
for filename in os.listdir(PATIENT_FOLDER_PATH):
    num_files += 1
    file_path = PATIENT_FOLDER_PATH + '/' + filename
    patient_df = pd.read_csv(file_path, na_values=['', '-'])

    name = filename.strip('.csv')
    check, gaps_dict = checkLargeGaps(patient_df)
    if check:   #for files with large gaps
        large_gap_dfs.append(name)
        patient_df = imputeLargestChunk(patient_df, gaps_dict)
        #print(name)
        #print()

        if patient_df is not None:   #for files with enough information
            patient_dfs[name] = patient_df
    else:
        small_gap_dfs.append(name)
        patient_dfs[name] = imputeLinear(patient_df)

# Storing DataFrames -> .csv files
for name, df in patient_dfs.items():
    new_file_path = NEW_PATIENT_FOLDER_PATH + '/' + name + '.csv'
    df.to_csv(new_file_path, index=False)

# Processing demographic files -> DataFrames
new_patient_ids = list(patient_dfs.keys())

demographic_df = pd.read_csv(FOLDER_PATH + '/demographic_all.csv')
initial_demographic_num = demographic_df.shape[0]

valid_rows = demographic_df['ID'].isin(new_patient_ids)
demographic_df = demographic_df[valid_rows]
new_demographic_num = demographic_df.shape[0]

demographic_df.to_csv(NEW_FOLDER_PATH + '/demographic_all.csv', index=False)
demographic_df_part = demographic_df.drop(['BCODE', 'UID'], axis=1)
demographic_df_part.to_csv(NEW_FOLDER_PATH + '/demographic.csv', index=False)

# Output Summary
print('Initial Number of Files: ', num_files)
print('Large Chunk Files: ', len(large_gap_dfs))
print('Small Chunk Files: ', len(small_gap_dfs))
print('Number of Files Removed: ', initial_demographic_num - new_demographic_num)
print()
print('Total Number of Files Imputed: ', len(patient_dfs))


"""
The following imputation methods were briefly tested
(1) Mean : impute the mean of each column
(2) Forward Fill : forward fill then fill remaining missing data with the mean
(3) Backward Fill : backward fill then fill remaining missing data with the mean

The following code utilizes the different imputation techniques:
mean_dict = {}
    for col in df.columns:
        if col != 'date':
            mean_dict[col] = np.mean(df[col])
    
    df_mean = df.fillna(mean_dict)

    df_ffill = df.ffill()
    df_ffill = df_ffill.fillna(mean_dict)

    df_bfill = df.bfill()
    df_bfill = df_bfill.fillna(mean_dict)

The following code performs data visualization:
fig, ax = plt.subplots(2, 3, sharey=True)

for col in test_df.columns:
    if col != 'date':
        ax[0].plot(df_mean['date'], df_mean[col], label='mean')
        ax[1].plot(df_ffill['date'], df_ffill[col], label='ffill')
        ax[2].plot(df_bfill['date'], df_bfill[col], label='bfill')

#plt.legend()
plt.show()
"""
