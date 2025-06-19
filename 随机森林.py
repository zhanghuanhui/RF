import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("data_.xlsx")

data['sex'] = data['sex'].astype('object')
data['BC'] = data['BC'].astype('object')
data['noduls'] = data['noduls'].astype('object')

X = data.iloc[:, 1:12]
Y = data.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
                                   X,
                                   Y,
                                   test_size=0.3,
                                   random_state=99)

from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier(random_state=99)

rfc_clf.fit(X_train, Y_train)
rfc_clf.score(X_test, Y_test)


#导入 joblib 库，用于模型的保存和加载
import joblib
joblib.dump(rfc_clf, 'rf.pkl')











