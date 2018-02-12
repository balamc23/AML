import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('adult.data.csv', header=None)
df2 = pd.read_csv('adult.test.csv', skiprows=1, header=None)
comb_df = df.append(df2)
# print comb_df

# Only selecting the columns corresponding to continuous attributes
comb_df = comb_df.ix[:, (0,2,4,10,11,12)]
# print comb_df
# Just writing to csv to check this file against Bala and Keshav's manual csv
comb_df.to_csv('cont_attrs.csv', sep=',', header=None)

# Dropping rows with missing values
comb_df.dropna(axis=0)
comb_df.to_csv('missdrop_cont_attrs.csv', sep=',', header=None)
# print comb_df

# Scaling each column to have unit variance
# df = pd.DataFrame(preprocessing.MinMaxScaler.fit_transform(df.T),
#              columns=df.columns, index=df.index)
comb_df = normalized_df=(comb_df-comb_df.min())/(comb_df.max()-comb_df.min())

# print(comb_df)