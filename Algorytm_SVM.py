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

dane_rzeczywiste=pd.read_csv('Iris.csv') #załadowanie danych z exella, wyswietlenie ich stat, usuniecie columny id
#print(dane_rzeczywiste)
#print(dane_rzeczywiste.describe())
dane_rzeczywiste.drop(columns = ['Id'], axis=1, inplace=True)
#print(dane_rzeczywiste)

# definiujemy SVM liniowego
class aSVM(BaseEstimator):
    def __init__(self, learning_rate=0.005, lambda_param=0.01, n_iters=1000): #inicjalizacja obiektu
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None #przechowywanie wag
        self.b = None #przechowywanie obciazenia

    def fit(self, X,y): #trenowanie modelu
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features) #self.w - wektor zer o dlugosci n_features
        self.b = 0 #obiciazenie = 0, obciażenie

        for _ in range(self.n_iters):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i],self.w) - self.b) >=1: # czy wynik klasyfikacji jest wiekszy od 1
                    self.w -= self.lr * (2 * self.lambda_param * self.w) #if poprawna klasyfikacja, aktualizauje sie waga
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w -np.dot(X[i],y[i])) #if niepoprawna klasyfikacja mniejsza od 1, to aktualizacja wagi o obciążenia
                    self.b -= self.lr * y[i]
    def predict(self, X): #przewiduje klas dla nowych X na podstawie self.w i self.b
        pred = np.sign(np.dot(X, self.w) - self.b) #sign zwraca 1 dla dodatnych lub -1 dla ujemnych, wynik to przewidywanie etykiet
        return pred

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
acc = accuracy_score(y_svm_test_test,y_pred)
print('Accurancy liniowego SVM z biblioteki: ', acc)

svc2 = aSVM()
svc2.fit(X_svm_test_train, y_svm_test_train)
y_pred2 = svc2.predict(X_svm_test_test)
acc2 = accuracy_score(y_svm_test_test, y_pred)
print('Accurancy zdefiniowanego liniowego SVM', acc2)
############################################################################################
print('dane rzeczywiste: ', y_rz)
print('dane syntetyczne: ', y)
#One vs One, N(N-1)/2
class one_vs_one(BaseEstimator): #cos nie chce dzialac
    def __init__(self):
        self.classifiers = []
    def fit(self, X, y):
        self.classes = np.unique(y).flatten()

        for class_a, class_b in combinations(self.classes, 2): #wybiera wszystkie możliwe pary
            szablon = (y==class_a) | (y == class_b) #True jeśli należą do jednej z dwóch klas ('a' lub 'b') jak nie to False
            para_X = X[szablon]
            para_y = y[szablon]

            classif = aSVM()
            classif.fit(para_X, para_y)
            self.classifiers.append(class_a,class_b, classif).flatten('C') #dodaje do arraya classifiers
    def predict(self, X): #glosowanie wiekszosciowe ta klasa co ma najwiecej glosow wygrywa, iteracja po all binarnych klasyfikatorach wytrenowanych na parach klas i zbieramy predykcje wskazujaca na konkretna klase
        przewidywane_etykiet = []

        for probka in X_test:
            class_votes = [0] * len(self.classes)

            for class_a, class_b, classif in self.classifiers:
                glos = classif.decision_function([probka])[0] #tu chyba nie może być decision_function bo jest zdefiniowana sklearn SVM ale idk

                if glos >0:
                    class_votes[class_a] +=1
                else:
                    class_votes[class_b] +=1
                przewidywane_etykiety = np.armax(class_votes) #wybiera klase z max,najwieksza iloscia glosow
                przewidywane_etykiet.append(przewidywane_etykiety) #etkieta dodawana do przewidywania etykiet
        return przewidywane_etykiet
    ######ONE VS REST##########
class one_vs_all(BaseEstimator):
    def __init__(self):
        self.classifiers = []

    def fit(self, X, y):
        self.classes = np.unique(y).flatten()




    ###### eksperymenty dla danych syntetycznych OVO #########
ovo = one_vs_one()
ovo.fit(X_train, y_train)
y_pred3 = ovo.predict(X_test)
acc3 = accuracy_score(y_test, y_pred3)
precision = precision_score(y_test, y_pred3)
recall = recall_score(y_test, y_pred3)
f1 = f1_score(y_test, y_pred3)
print(acc3, precision,recall, f1) #

np.save('wyniki_dane_syntetyczne.npy', [acc3, precision, recall, f1])
    ###### eksperymenty dla danych rzeczywistych OVO #########
ovo = one_vs_one()
ovo.fit(X_rz_train, y_rz_train)
y_pred4 = ovo.predict(X_rz_test)
acc4 = accuracy_score(y_rz_test, y_pred4)
precision2 = precision_score(y_rz_test, y_pred4)
recall2 = recall_score(y_rz_test, y_pred4)
f1_2 = f1_score(y_rz_test, y_pred3)
print(acc4, precision2, recall2, f1_2)  #

np.save('wyniki_dane_syntetyczne.npy', [acc4, precision2, recall2, f1_2])










