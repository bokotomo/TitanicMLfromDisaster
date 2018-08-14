"""
決定木分析によって予測をする
"""
import sys
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+'/'
sys.path.append(CURRENT_SCRIPT_PATH+'../')
import titanic_data
from sklearn.tree import DecisionTreeClassifier

def convert_train(titanic_data, keys=[]):
    """
    トレーニングデータの整形
    """
    y = []
    data = []

    # 欠損の含まれる行を削除
    titanic_data = titanic_data.dropna(subset=['Pclass', 'SibSp', 'Parch'])

    for k,v in titanic_data.iterrows():
        # 欠損を補完
        v = supplemente_missing_values(keys, v)

        x = [v[key] for key in keys]
        data.append(x)
        y.append(v['Survived'])

    return data, y

def convert_test(titanic_data, keys=[]):
    """
    テストデータの整形
    """
    data = []

    for k,v in titanic_data.iterrows():
        # 欠損を補完
        v = supplemente_missing_values(keys, v)

        # 欠損を0に変換
        for key in keys:
            if v[key] == None or str(v[key]).lower() == "null" or str(v[key]).lower() == "nan":
                v[key] = 0

        x = [v[key] for key in keys]
        data.append(x)

    return data

def supplemente_missing_values(keys, v):
    """
    欠損を補完
    """
    if "Age" in keys:
        if v["Age"] == None or str(v["Age"]).lower() == "null" or str(v["Age"]).lower() == "nan":
            if "master" in v["Name"].lower():
                v["Age"] = 5
            elif "miss" in v["Name"].lower():
                v["Age"] = 18
            elif "mrs" in v["Name"].lower():
                v["Age"] = 28
            else:
                v["Age"] = 30
    if "Sex" in keys:
        if v["Sex"] == None or str(v["Sex"]).lower() == "null" or str(v["Sex"]).lower() == "nan":
            if "miss" in v["Name"].lower():
                v["Sex"] = 1
            elif "mrs" in v["Name"].lower():
                v["Sex"] = 1
            elif "mr" in v["Name"].lower():
                v["Sex"] = 0
            else:
                v["Sex"] = 0

    return v

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

def main(args):
    """
    データ取得
    """
    train_df = titanic_data.get_train()
    test_df = titanic_data.get_test()

    """
    整形
    """
    # 要素一覧
    keys_paterns = [
        ['Age', 'Fare', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked'],
        ['Age', 'Sex', 'Pclass', 'SibSp', 'Parch'],
    ] 
    # 学習用に整形
    train, y = convert_train(train_df, keys=keys_paterns[1])
    test = convert_test(test_df, keys=keys_paterns[1])

    """
    学習
    """
    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(train, y)

    """
    予測
    """
    predict = clf.predict(test)

    """
    表示
    """
    show_result(test_df, predict)

    """
    CSVで書き出し
    """
    titanic_data.to_csv(test_df, predict)

if __name__ == '__main__':
    main(sys.argv)

