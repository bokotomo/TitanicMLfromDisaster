"""
ロジスティック回帰によって予測をする
"""
import sys
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+'/'
sys.path.append(CURRENT_SCRIPT_PATH+'../')
import titanic_manager
from sklearn.linear_model import LogisticRegression

def convert_train(titanic_manager, keys=[]):
    """
    トレーニングデータの整形
    """
    y = []
    data = []

    # 欠損の含まれる行を削除
    titanic_manager = titanic_manager.dropna(subset=['Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    for k,v in titanic_manager.iterrows():
        x = [v[key] for key in keys]
        data.append(x)
        y.append(v['Survived'])

    return data, y

def convert_test(titanic_manager, keys=[]):
    """
    テストデータの整形
    """
    data = []

    # 欠損を0に変換
    titanic_manager = titanic_manager.fillna(0)

    for k,v in titanic_manager.iterrows():
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
        ['Age', 'Sex', 'Pclass', 'SibSp', 'Parch'],
    ] 
    # 学習用に整形
    train, y = convert_train(train_df, keys=keys_paterns[1])
    test = convert_test(test_df, keys=keys_paterns[1])

    """
    学習
    """
    clf = LogisticRegression()
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

