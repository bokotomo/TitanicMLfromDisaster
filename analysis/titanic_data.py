import pandas
import numpy
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+"/"

def get_train():
    """
    訓練データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/train.csv', encoding="UTF-8")
    df = df.replace('male', 0).replace('female', 1).replace("S",0).replace("C",1).replace("Q",2) 
    print("traning")
    print(df.describe())
    return df

def get_test():
    """
    入力データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/test.csv', encoding="UTF-8")
    df = df.replace('male', 0).replace('female', 1).replace("S",0).replace("C",1).replace("Q",2)
    print("test")
    print(df.describe())
    return df

def get_gender_submission():
    """
    入力データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/gender_submission.csv', encoding="UTF-8")
    print("gender_submission")
    print(df.describe())
    return df

def to_csv(test, predict):
    """
    CSVで書き出し
    """
    PassengerId = numpy.array(test["PassengerId"]).astype(int)
    df = pandas.DataFrame(predict, PassengerId, columns = ["Survived"])
    df.to_csv("./result.csv", index_label = ["PassengerId"])

