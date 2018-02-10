import pandas as pd

df = pd.read_csv('adult.data.csv', header=None)

# Only selecting the columns corresponding to continuous attributes
df = df.ix[:, (0,2,4,10,11,12)]
print df
df.to_csv('cont_attrs.csv', sep=',', header=None)

# Dropping rows with missing values
df.dropna(axis=0)
df.to_csv('missdrop_cont_attrs.csv', sep=',', header=None)
print df
