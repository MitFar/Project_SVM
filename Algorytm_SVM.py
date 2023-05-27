import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import combinations
from algorytmy import *

dane_rzeczywiste=pd.read_csv('Iris.csv') #załadowanie danych z exella, wyswietlenie ich stat, usuniecie columny id
dane_rzeczywiste.drop(columns = ['Id'], axis=1, inplace=True)

y_rz = dane_rzeczywiste['Species']
X_rz = dane_rzeczywiste.drop(['Species'], axis=1)

etykiety = LabelEncoder() #zmiana danych tekstowych na dane liczbowe: 0,1,2
y_rz=etykiety.fit_transform(y_rz)
scaler = StandardScaler() #skaluje atrybuty
X_rz = scaler.fit_transform(X_rz)

X, y = make_classification(n_samples=150, n_informative=3, n_classes=3, n_features=4,n_redundant=1,  random_state=5) #dane syntetyczne
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) #podział danych syntetycznych
X_rz_train, X_rz_test, y_rz_train, y_rz_test = train_test_split(X_rz, y_rz, test_size=0.3, random_state=4) #podział danych rzezywistych

#TESTOWANIE CZY SVM DAJE TAKIE SAME WYNIKI JAK TEN Z BIBLIOTEKI
X_svm_test, y_svm_test = make_classification(n_samples=150, n_informative=2, n_classes=2, n_features=4,n_redundant=1,  random_state=50) #dane testowe dla liniowego SVM
X_svm_test_train, X_svm_test_test, y_svm_test_train, y_svm_test_test = train_test_split(X_svm_test, y_svm_test, test_size=0.3, random_state=4) # podzial na train i test

svc = SVC()
svc.fit(X_svm_test_train, y_svm_test_train) #uczenie
y_pred=svc.predict(X_svm_test_test)
acc = accuracy_score(y_svm_test_test, y_pred)
print('Accurancy liniowego SVM z biblioteki: ', acc)

svc2 = aSVM()
svc2.fit(X_svm_test_train, y_svm_test_train)
y_pred2 = svc2.predict(X_svm_test_test)
acc2 = accuracy_score(y_svm_test_test, y_pred)
print('Accurancy zdefiniowanego liniowego SVM', acc2)
print('dane rzeczywiste: ', y_rz)
print('dane syntetyczne: ', y)

    ###### eksperymenty dla danych syntetycznych OVO i OVA #########
ovo = one_vs_one()
ovo.fit(X_train, y_train)
y_pred3 = ovo.predict(X_test)
acc3 = accuracy_score(y_test, y_pred3)
print("acc score ovo syn: ", acc3)
print("recall score ovo syn: ", recall_score(y_test, y_pred3, average='macro',  zero_division=1))
print("prec score ovo syn: ", precision_score(y_test, y_pred3, average='macro', zero_division=1))
print("f1_score score ovo syn: ", f1_score(y_test, y_pred3, average='macro'))

ova = one_vs_all()
ova.fit(X_train,y_train)
y_pred_ova_syn = ova.predict(X_test)
acc_ova_syn = accuracy_score(y_test, y_pred_ova_syn)
print("\nOVA syntetyczne acc: ", acc_ova_syn)
print("recall score OVA syn: ", recall_score(y_test, y_pred_ova_syn, average='macro',  zero_division=1))
print("prec score OVA syn: ", precision_score(y_test, y_pred_ova_syn, average='macro', zero_division=1))
print("f1_score score OVA syn: ", f1_score(y_test, y_pred_ova_syn, average='macro'))

ovo = one_vs_one()
ovo.fit(X_rz_train, y_rz_train)
y_pred4 = ovo.predict(X_rz_test)
acc4 = accuracy_score(y_rz_test, y_pred4)
print("\n Ove-vs-one dane rzeczywiste\nOVO dane rz acc score: ", acc4)
print("recall score OVO syn: ", recall_score(y_rz_test, y_pred4, average='macro',  zero_division=1))
print("prec score OVO syn: ", precision_score(y_rz_test, y_pred4, average='macro', zero_division=1))
print("f1_score score OVO syn: ", f1_score(y_rz_test, y_pred4, average='macro'))

ova = one_vs_all()
ova.fit(X_rz_train, y_rz_train)
y_pred_ova_rz = ova.predict(X_rz_test)
acc_ova_rz = accuracy_score(y_rz_test, y_pred_ova_rz)
print("\n Ove-vs-all dane rzeczywiste\nOVA rzeczywiste acc score: ", acc_ova_rz)
print("recall score OVA rz: ", recall_score(y_rz_test, y_pred_ova_rz, average='macro',  zero_division=1))
print("prec score OVA rz: ", precision_score(y_rz_test, y_pred_ova_rz, average='macro', zero_division=1))
print("f1_score score OVA rz: ", f1_score(y_rz_test, y_pred_ova_rz, average='macro'))

###WALIDACJA KRZYŻOWA###

###TEST T-STUDENTA###

###WYKRESIKI TABELKI###