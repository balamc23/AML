import pandas as pd
import numpy as np

df = pd.read_csv('adult.data.csv', header=None)
df2 = pd.read_csv('adult.test.csv', skiprows=1, header=None)
comb_df = df.append(df2)
# print comb_df

# Only selecting the columns corresponding to continuous attributes
comb_df = comb_df.ix[:, (0,2,4,10,11,12)]
print comb_df
# Just writing to csv to check this file against Bala and Keshav's manual csv
comb_df.to_csv('cont_attrs.csv', sep=',', header=None)

# Dropping rows with missing values
comb_df.dropna(axis=0)
comb_df.to_csv('missdrop_cont_attrs.csv', sep=',', header=None)
# print comb_df

# Scaling each column to have unit variance
comb_df = (comb_df-comb_df.min())/(comb_df.max()-comb_df.min())
print(comb_df)

# Splitting the dataset into 80% training, 10% validation, and 10% testing
train_set, validate_set, test_set = np.split(comb_df.sample(frac = 1), [int(0.8*len(comb_df)), int(0.9*len(comb_df))])
print(len(comb_df))
print(len(train_set))
print(len(validate_set))
print(len(test_set))

# SVM code below
reg_consts = [1e-3, 1e-2, 1e-1, 1]
print(reg_consts[0], reg_consts[1], reg_consts[2], reg_consts[3])

# Hinge Loss Function
predicted_label = a_T*x + b
curr_cost = max(0, 1 - true_label*predicted_label)