# venvはpython3で、仮想環境を作成するコマンド
基本的にpip installは、仮想環境で行うべき。

# venvの環境を作成する ( python -m venv 環境の名前 )
python -m venv titanic

# 仮想環境に入る
source ./titanic/bin/activate

# 仮想環境から出る
deactivate

# メモ  
エイリアスを作成しておくと便利  
「~/.bash_profileファイル」に下記コード記入  
```
alias titanic="source /Users/名前/このスクリプトのPATH/venv/titanic/bin/activate"
```

## 変更内容の適用  
```
source ~/.bash_profile  
```

## 実行方法 (下のコマンドで仮想環境に入れる)  
```
$ titanic  
```

