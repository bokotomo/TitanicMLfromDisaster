"""
ランダムフォレストによって予測をする
"""
import sys
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+'/'
sys.path.append(CURRENT_SCRIPT_PATH+'../')
import titanic_manager
from sklearn.ensemble import RandomForestClassifier

def convert_train(train_df, keys=[]):
    """
    トレーニングデータの整形
    """
    y = []
    data = []

    for k,v in train_df.iterrows():
        x = [v[key] for key in keys]
        data.append(x)
        y.append(v['Survived'])

    return data, y

def convert_test(test_df, keys=[]):
    """
    テストデータの整形
    """
    data = []

    for k,v in test_df.iterrows():
        x = [v[key] for key in keys]
        data.append(x)

    return data

def main(args):
    """
    データ取得
    """
    train_df = titanic_manager.get_train()
    test_df = titanic_manager.get_test()

    """
    整形
    """
    # 要素一覧
    keys_paterns = [
        ['Age', 'Fare', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked'],
        ['Age', 'Sex', 'Pclass', 'IsAlone', 'FamSize', 'Fare', 'Title'],
        ['Age', 'Sex', 'Pclass', 'IsAlone', 'FamSize', 'Title'],
        ['Age', 'Sex', 'Pclass', 'SibSp', 'Parch'],
    ] 
    # 学習用に整形
    train, y = convert_train(train_df, keys=keys_paterns[1])
    test = convert_test(test_df, keys=keys_paterns[1])

    """
    学習
    """
    clf = RandomForestClassifier(n_estimators=10000)
    clf.fit(train, y)

    """
    予測
    """
    predict = clf.predict(test)

    """
    表示
    """
    titanic_manager.show_result(test_df, predict)

    """
    CSVで書き出し
    """
    titanic_manager.to_csv(test_df, predict)

if __name__ == '__main__':
    main(sys.argv)

