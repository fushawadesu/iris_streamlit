#基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

#データセットの読み込み
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#目標値
df['target'] = iris.target

#目標値を数字から花の名前に変更
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'

#予測モデルの構築
x = iris.data[:,[0,2]]
y = iris.target

#ロジスティクス回帰
clf = LogisticRegression()
clf.fit(x,y)

#サイドバー（入力画面）
st.sidebar.header('Input Features')

#スライダーの入力欄をサイドバーに設定した上、アヤメのがく片の長さと花弁の長さの値を設定できるように実装します。
#最小値を0cm、最大値を10cmとし、0.1cm刻みで値を変えられるように設定していきます。
sepalValue = st.sidebar.slider('sepal length(cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length(cm)', min_value=0.0, max_value=10.0, step=0.1)

#出力画面をメインパネルに表示
#メインパネル
#タイトルをiris classifierとして設定
st.title('Iris Classifier')
st.write('## Input Value')

#インプットデータ（１行のデータフレーム）
value_df = pd.DataFrame([], columns=['data', 'sepal length(cm)', 'petal length(cm)'])
record = pd.Series(['data', sepalValue, petalValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index('data', inplace=True)

#入力値の値を表示 #write methodを使用してね
st.write(value_df)

#出力されていた値を表示できるように予測値のデータフレームを実装
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=['setosa', 'versicoler','virginica'], index=['probability'])

st.write('## Prediction')
st.write(pred_df)

#予測結果の出力
#サイドバーで入力した値を使い、予測悔過を確率で出力させることができる。
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと', str(name[0]), 'です！')

