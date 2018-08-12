import sys
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy

def get_train():
    df = pd.read_csv('./data/train.csv', encoding="UTF-8")
    df = df.replace('male', 0).replace('female', 1).replace("S",0).replace("C",1).replace("Q",2) 
    print(df.mean())
    return df

def get_test():
    df = pd.read_csv('./data/test.csv', encoding="UTF-8")
    df = df.replace('male', 0).replace('female', 1).replace("S",0).replace("C",1).replace("Q",2)
    print(df.mean())
    return df

def main(args):
    df = get_train()
    # 
    #seaborn.pairploat( df[[ "Sex", "Survived", ]])
    # 散布図
    #seaborn.jointplot("Sex", "Survived", df, dropna=True)
    # 合計
    seaborn.barplot(x="Sex", y="Survived", data=df, estimator=sum)
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

