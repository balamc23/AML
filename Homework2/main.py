import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('adult.data.csv', header=None)
df2 = pd.read_csv('adult.test.csv', skiprows=1, header=None)
comb_df = df.append(df2)
labels_df = comb_df.ix[:, 14]
labels = labels_df.as_matrix()
for i in range((len(labels))):
    if (labels[i][1] == '<'):
        labels[i] = -1
    else:
        labels[i] = 1

labels_df = pd.DataFrame(labels)

# Only selecting the columns corresponding to continuous attributes
comb_df = comb_df.ix[:, (0,2,4,10,11,12)]

# Dropping rows with missing values
comb_df.dropna(axis=0)
comb_df.to_csv('missdrop_cont_attrs.csv', sep=',', header=None)

# Scaling each column to have unit variance
comb_df = (comb_df-comb_df.min())/(comb_df.max()-comb_df.min())

# Combining features df with labels df
comb_df = comb_df.reset_index(drop=True)
labels_df = labels_df.reset_index(drop=True)
comb_df = pd.concat([comb_df, labels_df], axis=1)

# Splitting the dataset into 80% training, 10% validation, and 10% testing
train_set, validate_set, test_set = np.split(comb_df.sample(frac = 1), [int(0.8*len(comb_df)), int(0.9*len(comb_df))])

# SVM code below
reg_consts = [1e-3, 1e-2, 1e-1, 1]

# SGD below
num_features = len(comb_df.columns) - 1
num_steps = 300
step_size = float(1)/num_steps
num_epochs = 50
validate_set = validate_set.values
num_reg_consts = len(reg_consts)
sums = np.zeros(num_reg_consts)
acc_arrs = []
coeff_arrs = []

# Finding the best regularization constant
batch_size = 100

for reg_ct, reg_const in enumerate(reg_consts):
    w = np.zeros(num_features)
    print('For regularization constant value of', reg_const, 'the accuracy values are as follows:')
    accs = []
    coeffs = []
    for i in range(num_epochs):
        eval_examples = train_set.sample(n=50).values
        for step_count in range(num_steps):
            sample_batch = validate_set[np.random.choice(validate_set.shape[0], batch_size, replace=False), :]
            for row in sample_batch:
                row = list(row)
                x = row[0:-1]
                y = row[-1]

                if (y*np.dot(x, w)) < 1:
                    w = w + step_size * (np.multiply(x, y) + (-2*reg_const*w))
                else:
                    w = w + step_size*(-2*reg_const*w)

            # Evaluating the classifier on the set held out for evaluation
            if ((step_count+1) % 30) == 0:
                curr_acc = 0
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
                curr_acc = float(num_correct)/len(eval_examples)
                sums[reg_ct] += curr_acc
                accs.append(curr_acc)
                coeffs.append(np.linalg.norm(w))
                print('Accuracy for epoch', i+1, 'at step', step_count+1, 'is', curr_acc)

    acc_arrs.append(accs)
    coeff_arrs.append(coeffs)
    print('=======================================================================================')


x_pts = np.asarray([i*30 for i in range(1,501)])
acc_arrs = np.asarray(acc_arrs)

plt.figure(1)
plt.plot(x_pts, acc_arrs[0])
plt.plot(x_pts, acc_arrs[1])
plt.plot(x_pts, acc_arrs[2])
plt.plot(x_pts, acc_arrs[3])
plt.legend(['1e-3', '1e-2', '1e-2', '1e-2'])
plt.xlim(0, 15000)
plt.xlabel('Number of steps')
plt.ylabel('Accuracy')
plt.show()

plt.figure(2)
plt.plot(x_pts, coeff_arrs[0])
plt.plot(x_pts, coeff_arrs[1])
plt.plot(x_pts, coeff_arrs[2])
plt.plot(x_pts, coeff_arrs[3])
plt.legend(['1e-3', '1e-2', '1e-2', '1e-2'])
plt.xlim(0, 15000)
plt.xlabel('Number of steps')
plt.ylabel('Coefficient Magnitude')
plt.show()

max_ind = 0
for i,elem in enumerate(sums):
    average = (elem/num_epochs)
    if(average > sums[max_ind]):
        max_ind = i

final_reg_const = reg_consts[max_ind]
print(final_reg_const)


# Training our best reg const SVM
train_set = train_set.values
w = np.zeros(num_features)
batch_size = 100

for i in range(num_epochs):
    for step_count in range(num_steps):
        print('Training classifier on epoch number:', i, 'for step number:', step_count)
        sample_batch = train_set[np.random.choice(train_set.shape[0], batch_size, replace=False), :]
        for row in sample_batch:
            row = list(row)
            x = row[0:-1]
            y = row[-1]

            if (y*np.dot(x, w)) < 1:
                w = w + step_size * (np.multiply(x, y) + (-2*final_reg_const*w))
            else:
                w = w + step_size*(-2*final_reg_const*w)


# Evaluating our SVM on the testing set
test_set = test_set.values
num_correct = 0

for row in test_set:
    row = list(row)
    x = row[0:-1]
    y = row[-1]

    pred = np.dot(w, x)

    if pred > 0 and y == 1:
        num_correct += 1
    elif pred < 0 and y == -1:
        num_correct += 1

accuracy = float(num_correct)/len(test_set)

print('Accuracy of our classifier is', (accuracy))


