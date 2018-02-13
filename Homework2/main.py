import pandas as pd
import numpy as np

df = pd.read_csv('adult.data.csv', header=None)
df2 = pd.read_csv('adult.test.csv', skiprows=1, header=None)
comb_df = df.append(df2)
labels_df = comb_df.ix[:, 14]
labels = labels_df.as_matrix()
for i in range((len(labels))):
    if(labels[i][1] == '<'):
        labels[i] = -1
    else:
        labels[i] = 1

# print comb_df
# print(labels_df)
# print(labels)
labels_df = pd.DataFrame(labels)
# print(labels_df)



# Only selecting the columns corresponding to continuous attributes
comb_df = comb_df.ix[:, (0,2,4,10,11,12)]
# comb_df.add(labels_df,axis='columns')
# comb_df.insert(0,labels_df,series)
# print(comb_df)


# print comb_df
# Just writing to csv to check this file against Bala and Keshav's manual csv
comb_df.to_csv('cont_attrs.csv', sep=',', header=None)

# Dropping rows with missing values
comb_df.dropna(axis=0)
comb_df.to_csv('missdrop_cont_attrs.csv', sep=',', header=None)
# print comb_df

# Scaling each column to have unit variance
comb_df = (comb_df-comb_df.min())/(comb_df.max()-comb_df.min())
# print(comb_df)

comb_df = comb_df.reset_index(drop=True)
labels_df = labels_df.reset_index(drop=True)
comb_df = pd.concat([comb_df, labels_df], axis=1)
# print(comb_df)

# Splitting the dataset into 80% training, 10% validation, and 10% testing
train_set, validate_set, test_set = np.split(comb_df.sample(frac = 1), [int(0.8*len(comb_df)), int(0.9*len(comb_df))])
# print(len(comb_df))
# print(len(train_set))
# print(len(validate_set))
# print(len(test_set))

# SVM code below
# reg_consts = [1e-3, 1e-2, 1e-1, 1]
reg_consts = [1e-3]
# print(reg_consts[0], reg_consts[1], reg_consts[2], reg_consts[3])

# Hinge Loss Function
total_cost = 0

# for example in examples:
#     predicted_label = a_T*x + b
#     curr_cost = max(0, 1 - true_label*predicted_label)
#     total_cost += curr_cost
#
# hinge_loss = total_cost/len(examples) + reg_const*((a_T * a)/2)


# SGD below
num_features = len(comb_df.columns) - 1
w = np.zeros(num_features)
num_steps = 300
step_size = float(1)/num_steps
num_epochs = 50
validate_set = validate_set.values
# print(validate_set)

for reg_const in reg_consts:
    for i in range(num_epochs):
        for row in validate_set:
            row = list(row)
            x = row[0:-1]
            y = row[-1]
            # print(x)
            # print(y)

            if (y*np.dot(x, w)) < 1:
                w = w + step_size * (np.multiply(x, y) + (-2*reg_const*w))
            else:
                w = w + step_size*(-2*reg_const*w)

        eval_examples = train_set.sample(n=50).values
        num_correct = 0
        for row in eval_examples:
            row = list(row)
            x = row[0:-1]
            y = row[-1]

            pred = np.dot(w, x)

            if pred > 0 and y == 1:
                num_correct += 1
            elif pred < 0 and y == -1:
                num_correct += 1

        print('Accuracy for epoch', i+1, 'is', float(num_correct)/len(eval_examples))

# test_ex = list(validate_set[0])
# print(test_ex)
#
# prod = np.dot(test_ex, w)
# print(prod)









