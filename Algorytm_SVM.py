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
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from algorytmy import *
from sklearn.naive_bayes import GaussianNB

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
#print('dane rzeczywiste: ', y_rz)
#print('dane syntetyczne: ', y)

    ###### eksperymenty dla danych syntetycznych OVO i OVA #########
ovo = one_vs_one()
ovo.fit(X_train, y_train)
y_pred3 = ovo.predict(X_test)
acc3 = accuracy_score(y_test, y_pred3)
print("acc score ovo syn: ", acc3)
print("recall score ovo syn: ", recall_score(y_test, y_pred3, average='macro',  zero_division=0)) #zero div definiuje wartosc jaka ma byc zwrocona jesli zajdzie dzielenie przez 0
print("prec score ovo syn: ", precision_score(y_test, y_pred3, average='macro', zero_division=0))
print("f1_score score ovo syn: ", f1_score(y_test, y_pred3, average='macro', zero_division=0))

ova = one_vs_all()
ova.fit(X_train,y_train)
y_pred_ova_syn = ova.predict(X_test)
acc_ova_syn = accuracy_score(y_test, y_pred_ova_syn)
print("\nOVA syntetyczne acc: ", acc_ova_syn)
print("recall score OVA syn: ", recall_score(y_test, y_pred_ova_syn, average='macro'))
print("prec score OVA syn: ", precision_score(y_test, y_pred_ova_syn, average='macro'))
print("f1_score score OVA syn: ", f1_score(y_test, y_pred_ova_syn, average='macro'))
print('Y_testy dla ovo syn',y_test)
print('Y pred ova syn', y_pred_ova_syn)

ovo = one_vs_one()
ovo.fit(X_rz_train, y_rz_train)
y_pred4 = ovo.predict(X_rz_test)
acc4 = accuracy_score(y_rz_test, y_pred4)
print("\n Ove-vs-one dane rzeczywiste\nOVO dane rz acc score: ", acc4)
print("recall score OVO syn: ", recall_score(y_rz_test, y_pred4, average='macro',  zero_division=0))
print("prec score OVO syn: ", precision_score(y_rz_test, y_pred4, average='macro', zero_division=0))
print("f1_score score OVO syn: ", f1_score(y_rz_test, y_pred4, average='macro', zero_division=0))

ova = one_vs_all()
ova.fit(X_rz_train, y_rz_train)
y_pred_ova_rz = ova.predict(X_rz_test)
acc_ova_rz = accuracy_score(y_rz_test, y_pred_ova_rz)
print("\n Ove-vs-all dane rzeczywiste\nOVA rzeczywiste acc score: ", acc_ova_rz)
print("recall score OVA rz: ", recall_score(y_rz_test, y_pred_ova_rz, average='macro'))
print("prec score OVA rz: ", precision_score(y_rz_test, y_pred_ova_rz, average='macro'))
print("f1_score score OVA rz: ", f1_score(y_rz_test, y_pred_ova_rz, average='macro'))




###WALIDACJA KRZYŻOWA###


rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
clfss= [one_vs_one(),one_vs_all(), OneVsOneClassifier(SVC(kernel='linear')), OneVsRestClassifier(SVC(kernel='linear')), GaussianNB(), KNeighborsClassifier()]

scores_syn = np.zeros((len(clfss),10, 4))


for i, (train_index, test_index) in enumerate(rskf.split(X_train, y_train)):
    X_train_rskf, y_train_rskf = X[train_index], y[train_index]
    X_test_rskf, y_test_rskf = X_train[test_index], y_train[test_index]
    for clf_id, base_clf in enumerate(clfss):
        clone_clf = clone(base_clf)
        clone_clf.fit(X_train_rskf, y_train_rskf)
        pred_rskf =clone_clf.predict(X_test_rskf)
        scores_syn[clf_id,i, 0] = accuracy_score(y_test_rskf, pred_rskf)
        print('y_test_rskf: ',y_test_rskf)
        print('pred_rsfk',pred_rskf)
        scores_syn[clf_id,i, 1] = precision_score(y_test_rskf, pred_rskf, average='macro')
        scores_syn[clf_id,i, 2] = recall_score(y_test_rskf, pred_rskf, average='macro')
        scores_syn[clf_id,i, 3] = f1_score(y_test_rskf, pred_rskf, average='macro')
print(scores_syn)
avg = np.average(scores_syn, axis=1)

#print(avg)
std = np.std(scores_syn, axis=1)


for clf_id, value in enumerate(clfss):
    print("\nDla", clfss[clf_id],  "  dane syntetyczne: ")
    print('Średnia accuracy:', avg[clf_id,0])
    print('Średnia precision:', avg[clf_id,1])
    print('Średnia recall:', avg[clf_id,2])
    print('Średnia F1-score:', avg[clf_id,3])
    print('Odchylenie standardowe accuracy:', std[clf_id,0])
    print('Odchylenie standardowe precision:', std[clf_id,1])
    print('Odchylenie standardowe recall:', std[clf_id,2])
    print('Odchylenie standardowe F1-score:', std[clf_id,3], "\n")


np.save('scores_syn.npy', scores_syn)

print("\n-------------------------")

scores_rz = np.zeros((len(clfss),10, 4))


for i, (train_index, test_index) in enumerate(rskf.split(X_rz_train, y_rz_train)):
    X_train_rskf, y_train_rskf = X[train_index], y[train_index]
    X_test_rskf, y_test_rskf = X_train[test_index], y_train[test_index]
    for clf_id, value in enumerate(clfss):
        clfss[clf_id].fit(X_train_rskf, y_train_rskf)

        pred_rskf =clfss[clf_id].predict(X_test_rskf)
        scores_rz[clf_id,i, 0] = accuracy_score(y_test_rskf, pred_rskf)
        scores_rz[clf_id,i, 1] = precision_score(y_test_rskf, pred_rskf, average='macro',zero_division=1)
        scores_rz[clf_id,i, 2] = recall_score(y_test_rskf, pred_rskf, average='macro',zero_division=1)
        scores_rz[clf_id,i, 3] = f1_score(y_test_rskf, pred_rskf, average='macro',zero_division=1)
#print(scores_rz)
avg_rz = np.average(scores_rz, axis=1)
#print("AVG" )
#print(avg_rz)
std_rz = np.std(scores_rz, axis=1)
np.save('scores_rz.npy', scores_rz)


for clf_id, value in enumerate(clfss):
    print("\nDla", clfss[clf_id],  "  dane rzeczywiste: ")
    print('Średnia accuracy:', avg_rz[clf_id,0])
    print('Średnia precision:', avg_rz[clf_id,1])
    print('Średnia recall:', avg_rz[clf_id,2])
    print('Średnia F1-score:', avg_rz[clf_id,3])
    print('Odchylenie standardowe accuracy:', std_rz[clf_id,0])
    print('Odchylenie standardowe precision:', std_rz[clf_id,1])
    print('Odchylenie standardowe recall:', std_rz[clf_id,2])
    print('Odchylenie standardowe F1-score:', std_rz[clf_id,3],"\n")









###TEST T-STUDENTA###

alpha = 0.05
metryki = ['Accuracy', 'Precision', 'Recall', 'F1']

for m_index in range(len(metryki)):

    for i in range(len(clfss)):
        for j in range(i + 1, len(clfss)):
            scores1 = scores_syn[i, :, m_index]
            scores2 = scores_syn[j, :, m_index]
            t_statistic, p_value = ttest_rel(scores1, scores2)

            print("Test t studenta ", metryki[m_index], " ", clfss[i]," i " ,clfss[j]," dane syntetyczne")
            print("Wartośść statystyki t ", t_statistic)
            print("Wartość p-value", p_value)
            if p_value > alpha:
                print('Brak istotnych roznic statystycznych')
            elif t_statistic > 0:
                print('Pierwszy algorytm jest statystyczne lepszy')
            else:
                print('Drugi algorytm jest statystycznie lepszy.\n')



for m_index in range(len(metryki)):

    for i in range(len(clfss)):
        for j in range(i + 1, len(clfss)):
            scores1 = scores_rz[i, :, m_index]
            scores2 = scores_rz[j, :, m_index]
            t_statistic, p_value = ttest_rel(scores1, scores2)

            print("Test t studenta ", metryki[m_index], " ", clfss[i]," i " ,clfss[j]," dane rzeczywiste")
            print("Wartośść statystyki t ", t_statistic)
            print("Wartość p-value", p_value)
            if p_value > alpha:
                print('Brak istotnych roznic statystycznych')
            elif t_statistic > 0:
                print('Pierwszy algorytm jest statystyczne lepszy')
            else:
                print('Drugi algorytm jest statystycznie lepszy.\n')

#Tabelki
from tabulate import tabulate

table_data = [
    ["One vs one dane syntetyczne", acc3, precision_score(y_test, y_pred3, average='macro', zero_division=0),recall_score(y_test, y_pred3, average='macro',  zero_division=0), f1_score(y_test, y_pred3, average='macro', zero_division=0)],
    ["One vs one dane rzeczywiste", acc4, precision_score(y_rz_test, y_pred4, average='macro', zero_division=0), recall_score(y_rz_test, y_pred4, average='macro',  zero_division=0), f1_score(y_rz_test, y_pred4, average='macro', zero_division=0)],
    ["One vs all dane syntetyczne", acc_ova_syn, precision_score(y_test, y_pred_ova_syn, average='macro', zero_division=0), recall_score(y_test, y_pred_ova_syn, average='macro',  zero_division=0), f1_score(y_test, y_pred_ova_syn, average='macro',zero_division=0)],
    ["One vs all dane rzeczywiste", acc_ova_rz, precision_score(y_rz_test, y_pred_ova_rz, average='macro', zero_division=0), recall_score(y_rz_test, y_pred_ova_rz, average='macro',  zero_division=0), f1_score(y_rz_test, y_pred_ova_rz, average='macro', zero_division=0)]
]


head = ["Metoda", "Accuracy", "Precision", "Recall", "F1"]


table = tabulate(table_data, headers=head, tablefmt="grid")


print("\n\n\nPrezentacja jakości metryk w formie tabeli przed przeprowadzeniem walidacji krzyżowej")

print(table)


table1_data = [
    ["One vs one dane syntetyczne", avg[0,0], avg[0,1], avg[0,2], avg[0,3]],
    ["One vs one dane rzeczywiste", avg_rz[0,0], avg_rz[0,1], avg_rz[0,2], avg_rz[0,3]],
    ["One vs all dane syntetyczne", avg[1,0], avg[1,1], avg[1,2], avg[1,3]],
    ["One vs all dane rzeczywiste", avg_rz[1,0], avg_rz[1,1], avg_rz[1,2], avg_rz[1,3]]
]


head1 = ["Metoda", "Accuracy", "Precision", "Recall", "F1"]


table1_data = [
    ["One vs one dane syntetyczne", avg[0,0],std[0,0], avg[0,1],std[0,1], avg[0,2],std[0,2], avg[0,3],std[0,3]],
    ["One vs one dane rzeczywiste", avg_rz[0,0],std_rz[0,0], avg_rz[0,1],std_rz[0,1], avg_rz[0,2],std_rz[0,2], avg_rz[0,3],std_rz[0,3]],
    ["One vs all dane syntetyczne", avg[1,0],std[1,0], avg[1,1],std[1,1], avg[1,2],std[1,2], avg[1,3],std[1,3]],
    ["One vs all dane rzeczywiste", avg_rz[1,0],std_rz[1,0], avg_rz[1,1],std_rz[1,1], avg_rz[1,2],std_rz[1,2], avg_rz[1,3],std_rz[1,3]],
    ["One vs one z biblio dane  syntetyczne", avg[2,0],std[2,0], avg[2,1],std[2,1], avg[2,2],std[2,2], avg[2,3],std[2,3]],
    ["One vs one z biblio dane rzeczywiste ", avg_rz[2,0],std_rz[2,0], avg_rz[2,1],std_rz[2,1], avg_rz[2,2],std_rz[2,2], avg_rz[2,3],std_rz[2,3]],
    ["One vs all z biblio dane syntetyczne ", avg[3,0],std[3,0], avg[3,1],std[3,1], avg[3,2],std[3,2], avg[3,3],std[3,3]],
    ["One vs all z biblio dane rzeczywiste ", avg_rz[3,0],std_rz[3,0], avg_rz[3,1],std_rz[3,1], avg_rz[3,2],std_rz[3,2], avg_rz[3,3],std_rz[3,3]],
    ["GNB z biblio dane syntetyczne ", avg[4,0],std[4,0], avg[4,1],std[4,1], avg[4,2],std[4,2], avg[4,3],std[4,3]],
    ["GNB z biblio dane rzeczywiste ", avg_rz[4,0],std_rz[4,0], avg_rz[4,1],std_rz[4,1], avg_rz[4,2],std_rz[4,2], avg_rz[4,3],std_rz[4,3]],
    ["KNN z biblio dane syntetyczne ", avg[5,0],std[5,0], avg[5,1],std[5,1], avg[5,2],std[5,2], avg[5,3],std[5,3]],
    ["KNN z biblio dane rzeczywiste ", avg_rz[5,0],std_rz[5,0], avg_rz[5,1],std_rz[5,1], avg_rz[5,2],std_rz[5,2], avg_rz[5,3],std_rz[5,3]]



]


head1 = ["Metoda", "Accuracy", "std Accuracy", "Precision", "std precision", "Recall","std recall", "F1", "std F1"]


table1 = tabulate(table1_data, headers=head1, tablefmt="grid")


print("\n\n\nPrezentacja jakości metryk w formie tabeli po przeprowadzeniu agregacji wyników z foldów walidacji krzyżowej")

print(table1)



###WYKRESIKI TABELKI###

categories = ['OVO  imp', 'Ova imp', 'OVO bib','OVA bib', "GNB", "kNN"]
data = [avg[0,0], avg[1,0],avg[2,0], avg[3,0],avg[4,0], avg[5,0]]
data1 = [avg[0,1], avg[1,1],avg[2,1], avg[3,1],avg[4,1], avg[5,1]]
data2 = [avg[0,2], avg[1,2],avg[2,2], avg[3,2],avg[4,2], avg[5,2]]
data3 = [avg[0,3], avg[1,3],avg[2,3], avg[3,3],avg[4,3], avg[5,3]]
std_dev = [std[0,0], std[1,0], std[2,0],std[3,0],std[4,0],std[5,0]]
std_dev1 = [std[0,1], std[1,1], std[2,1],std[3,1],std[4,1],std[5,1]]
std_dev2 = [std[0,2], std[1,2], std[2,2],std[3,2],std[4,2],std[5,2]]
std_dev3 = [std[0,3], std[1,3], std[2,3],std[3,3],std[4,3],std[5,3]]



fig, axs = plt.subplots(2, 2, figsize=(10, 8))



axs[0, 0].bar(categories, data, yerr=std_dev, capsize=10)
axs[0, 0].set_xlabel('Metody')
axs[0, 0].set_ylabel('Wartosc accuracy')
axs[0, 0].set_title('Wykres porownaie accuracy dla danych syntetycznych')

axs[0, 1].bar(categories, data1, yerr=std_dev1, capsize=10)
axs[0, 1].set_xlabel('Metody')
axs[0, 1].set_ylabel('Wartosc precsiom')
axs[0, 1].set_title('Wykres porownaie precisoion dla danych syntetycznych')

axs[1, 0].bar(categories, data2, yerr=std_dev2, capsize=10)
axs[1, 0].set_xlabel('Metody')
axs[1, 0].set_ylabel('Wartosc recall')
axs[1, 0].set_title('Wykres porownaie recall dla danych syntetycznych')

axs[1, 1].bar(categories, data3, yerr=std_dev3, capsize=10)
axs[1, 1].set_xlabel('Metody')
axs[1, 1].set_ylabel('Wartosc F1')
axs[1, 1].set_title('Wykres porownaie F1 dla danych syntetycznych')




plt.tight_layout()
plt.show()



categories1 = ['OVO  imp', 'Ova imp', 'OVO bib','OVA bib', "GNB", "kNN"]
ddata = [avg_rz[0,0], avg_rz[1,0],avg_rz[2,0], avg_rz[3,0],avg_rz[4,0], avg_rz[5,0]]
ddata1 = [avg_rz[0,1], avg_rz[1,1],avg_rz[2,1], avg_rz[3,1],avg_rz[4,1], avg_rz[5,1]]
ddata2 = [avg_rz[0,2], avg_rz[1,2],avg_rz[2,2], avg_rz[3,2],avg_rz[4,2], avg_rz[5,2]]
ddata3 = [avg_rz[0,3], avg_rz[1,3],avg_rz[2,3], avg_rz[3,3],avg_rz[4,3], avg_rz[5,3]]
std_ddev = [std_rz[0,0], std_rz[1,0], std_rz[2,0],std_rz[3,0],std_rz[4,0],std_rz[5,0]]
std_ddev1 = [std_rz[0,1], std_rz[1,1], std_rz[2,1],std_rz[3,1],std_rz[4,1],std_rz[5,1]]
std_ddev2 = [std_rz[0,2], std_rz[1,2], std_rz[2,2],std_rz[3,2],std_rz[4,2],std_rz[5,2]]
std_ddev3 = [std_rz[0,3], std_rz[1,3], std_rz[2,3],std_rz[3,3],std_rz[4,3],std_rz[5,3]]



fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))



axs1[0, 0].bar(categories1, ddata, yerr=std_ddev, capsize=10)
axs1[0, 0].set_xlabel('Metody')
axs1[0, 0].set_ylabel('Wartosc accuracy')
axs1[0, 0].set_title('Wykres porownaie accuracy dla danych rzeczywistych')

axs1[0, 1].bar(categories, ddata1, yerr=std_ddev1, capsize=10)
axs1[0, 1].set_xlabel('Metody')
axs1[0, 1].set_ylabel('Wartosc precsiom')
axs1[0, 1].set_title('Wykres porownaie precisoion dla danych rzeczywistych')

axs1[1, 0].bar(categories, ddata2, yerr=std_ddev2, capsize=10)
axs1[1, 0].set_xlabel('Metody')
axs1[1, 0].set_ylabel('Wartosc recall')
axs1[1, 0].set_title('Wykres porownaie recall dla danych rzeczywistych')

axs1[1, 1].bar(categories, ddata3, yerr=std_ddev3, capsize=10)
axs1[1, 1].set_xlabel('Metody')
axs1[1, 1].set_ylabel('Wartosc F1')
axs1[1, 1].set_title('Wykres porownaie F1 dla danych rzeczywistych')




plt.tight_layout()
plt.show()