"""
タイタニック問題のデータ取得やCSV吐き出しの処理をまとめたモジュール
"""
import pandas
import numpy
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+"/"
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def get_train():
    """
    訓練データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/train.csv', encoding="UTF-8")

    # 欠損を補完
    __supplemente_missing_values(df)

    # 欠損を変換
    df = df.fillna({'Embarked': 'S'})
    df = df.fillna(0)

    # 新しいカラムを作成
    __create_columns(df)

    print("------ TRANING ------")
    print(df.describe())
    return df

def get_test():
    """
    入力データの取得
    """
    df = pandas.read_csv(CURRENT_SCRIPT_PATH+'../data/test.csv', encoding="UTF-8")

    # 欠損を補完
    __supplemente_missing_values(df)

    # 欠損を変換
    df = df.fillna({'Embarked': 'S'})
    df = df.fillna(0)

    # 新しいカラムを作成
    __create_columns(df)

    print("------ TEST ------")
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

def show_result(test, predict):
    """
    一部表示
    """
    titles = {
        0: '死亡',
        1: '生存'
    }
    print('\n-------------------------------------------------------------')
    print('*  予測結果 *')
    print('-------------------------------------------------------------')
    for index,df in test[:5].iterrows():
        print(df)
        print(titles[predict[index]])
        print('-------------------------------------------------------------')

def __create_columns(df):
    """
    新しいカラムを作成
    """
    # 家族数の追加
    df["FamSize"] = df["SibSp"] + df["Parch"] + 1
    # 個人かどうかの追加
    df["IsAlone"] = df.FamSize.apply(lambda x: 1 if x == 1 else 0)
    # 名前の敬称の取得
    df["Title"] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    stat_min = 10
    title_names = (df['Title'].value_counts() < stat_min)
    df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    # 名前の敬称にONEHOT
    df['Title'] = LabelEncoder().fit_transform(df['Title'])
    # 性別にONEHOT
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    # 乗った場所にONEHOT
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

def __supplemente_missing_values(df):
    """
    欠損を補完
    """
    key = 'Age'
    for index,v in df[df[key].isnull()].iterrows():
        name = v["Name"].lower()
        if "master" in name:
            df[key].iat[index] = 5
        elif "miss" in name:
            df[key].iat[index] = 18
        elif "mrs" in name:
            df[key].iat[index] = 28
        else:
            df[key].iat[index] = 30

    key = 'Sex'
    for index,v in df[df[key].isnull()].iterrows():
        name = v["Name"].lower()
        if "miss" in name:
            df[key].iat[index] = 1
        elif "mrs" in name:
            df[key].iat[index] = 1
        elif "mr" in name:
            df[key].iat[index] = 0
        else:
            df[key].iat[index] = 0


