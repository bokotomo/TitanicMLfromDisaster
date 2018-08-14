import sys
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+'/'
sys.path.append(CURRENT_SCRIPT_PATH+'../')
import titanic_data
from sklearn.tree import DecisionTreeClassifier

def convert_train(titanic_data, keys):
    """
    トレーニングデータの整形
    """
    y = []
    data = []
    for k,v in titanic_data.iterrows():
        y.append(v['Survived'])
        x = [v[key] for key in keys]
        data.append(x)
    return data, y

def convert_test(titanic_data, keys):
    """
    テストデータの整形
    """
    data = []
    for k,v in titanic_data.iterrows():
        x = [v[key] for key in keys]
        data.append(x)
    return data

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
    train = titanic_data.get_train()
    test = titanic_data.get_test()

    """
    整形
    """
    # nullを0埋め
    train = train.fillna(0)
    test = test.fillna(0)
    # 要素一覧
    keys = ['Age', 'Fare', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
    # 学習用に整形
    train_data, y = convert_train(train, keys)
    test_data = convert_test(test, keys)

    """
    学習
    """
    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(train_data, y)

    """
    予測
    """
    predict = clf.predict(test_data)

    """
    表示
    """
    show_result(test, predict)

    """
    CSVで書き出し
    """
    titanic_data.to_csv(test, predict)

if __name__ == '__main__':
    main(sys.argv)

