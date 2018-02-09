import csv
import pandas as pd

# def load_data():

def main():
    filename = df.to_csv('adult_data.csv')
    x = pd.read_csv('adult_data.csv')
    print(x)

main()


# train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
