import sys
import seaborn
import matplotlib.pyplot as plt
import titanic

def pairplot(df=None, arr=[]):
    seaborn.pairplot(df[arr])
    plt.show()

def barplot(df=None, x="", y=""):
    # 散布図
    seaborn.barplot(x=x, y=y, data=df, estimator=sum)
    plt.show()

def main(args):
    train = titanic.get_train()
    test = titanic.get_test()
    pairplot(df=train, arr=["Sex", "Survived"])
    #barplot(df=train, x="Embarked", y="Survived")

if __name__ == "__main__":
    main(sys.argv)

