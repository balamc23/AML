import csv
import pandas as pd
import numpy as np

# def load_data():

def main():
    df = pd.read_csv('adult_data.csv', header = None)
    print(len(df))
    # train, validate, test = np.split(df.sample(frac = 1), [int(0.8*len(df)), int(0.9*len(df))])
    # print('train: ', len(train))
    # print('validate: ', len(validate))
    # print('test: ', len(test))

    # print(df)
main()


# train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
