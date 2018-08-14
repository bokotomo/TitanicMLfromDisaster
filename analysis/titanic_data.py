import pandas
import numpy
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+"/"

def get_train():
    """
    訓練データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/train.csv', encoding="UTF-8")
    # 文字列を数字に変換
    df = df.replace('male', 0).replace('female', 1).replace("S",0).replace("C",1).replace("Q",2)
    # 欠損を補完
    __supplemente_missing_values(df)
    print("traning")
    print(df.describe())
    return df

def get_test():
    """
    入力データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/test.csv', encoding="UTF-8")
    # 文字列を数字に変換
    df = df.replace('male', 0).replace('female', 1).replace("S",0).replace("C",1).replace("Q",2)
    # 欠損を補完
    __supplemente_missing_values(df)
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
    PassengerIds = numpy.array(test["PassengerId"]).astype(int)
    df = pandas.DataFrame(predict, PassengerIds, columns = ["Survived"])
    df.to_csv("./result.csv", index_label = ["PassengerId"])

def __supplemente_missing_values(df):
    """
    欠損を補完
    """

    key = 'Age'
    for index,v in df[df[key].isnull()].iterrows():
        if "master" in v["Name"].lower():
            df[key].iat[index] = 5
        elif "miss" in v["Name"].lower():
            df[key].iat[index] = 18
        elif "mrs" in v["Name"].lower():
            df[key].iat[index] = 28
        else:
            df[key].iat[index] = 30

    key = 'Sex'
    for index,v in df[df[key].isnull()].iterrows():
        if "miss" in v["Name"].lower():
            df[key].iat[index] = 1
        elif "mrs" in v["Name"].lower():
            df[key].iat[index] = 1
        elif "mr" in v["Name"].lower():
            df[key].iat[index] = 0
        else:
            df[key].iat[index] = 0

