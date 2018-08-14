import sys
from os import path
CURRENT_SCRIPT_PATH = path.dirname(path.abspath( __file__ ))+"/"
sys.path.append(CURRENT_SCRIPT_PATH+'../')
import titanic_data
from sklearn.tree import DecisionTreeClassifier

def convert_train(titanic_data, columns):
    """
    convert titanic_data.
    """
    y = []
    data = []
    for k,v in titanic_data.iterrows():
        y.append(v["Survived"])
        x = [v[column] for column in columns]
        data.append(x)
    return data, y

def convert_test(titanic_data, columns):
    """
    convert titanic_data.
    """
    data = []
    for k,v in titanic_data.iterrows():
        x = [v[column] for column in columns]
        data.append(x)
    return data

def show_resulta(test, predict):
    """
    """
    titles = {
        0: "死亡",
        1: "生存"
    }
    print("\n-------------------------------------------------------------")
    print("*  予測結果 *")
    print("-------------------------------------------------------------")
    for index,df in test[:5].iterrows():
        print(df)
        print(titles[predict[index]])
        print("-------------------------------------------------------------")

def main(args):
    """
    データ取得
    """
    train = titanic_data.get_train()
    test = titanic_data.get_test()

    """
    整形
    """
    train = train.fillna(0)
    test = test.fillna(0)
    columns = ["Age", "Fare", "Sex", "Pclass", "SibSp", "Parch", "Embarked"]
    train_data, y = convert_train(train, columns)
    test_data = convert_test(test, columns)

    """
    学習
    """
    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(train_data, y)
    predict = clf.predict(test_data)

    """
    表示
    """
    show_resulta(test, predict)

    """
    CSVで書き出し
    """
    titanic_data.to_csv(test, predict)

if __name__ == "__main__":
    main(sys.argv)

