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

        for k in range(self.n_iters):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i],self.w) - self.b) >=1: # czy wynik klasyfikacji jest wiekszy od 1
                    self.w -= self.lr * (2 * self.lambda_param * self.w) #if poprawna klasyfikacja, aktualizauje sie waga
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(X[i], y[i])) #if niepoprawna klasyfikacja mniejsza od 1, to aktualizacja wagi o obciążenia
                    self.b -= self.lr * y[i]
    def predict(self, X): #przewiduje klas dla nowych X na podstawie self.w i self.b
        pred = np.sign(np.dot(X, self.w) - self.b) #sign zwraca 1 dla dodatnych lub -1 dla ujemnych, wynik to przewidywanie etykiet
        return pred
#One vs One, N(N-1)/2
class one_vs_one(BaseEstimator): #
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

            self.classifiers.append((class_a, class_b, classif)) #dodaje do arraya classifiers
            #self.classifiers = np.array(self.classifiers).flatten('C')
    def predict(self, X_test): #glosowanie wiekszosciowe ta klasa co ma najwiecej glosow wygrywa, iteracja po all binarnych klasyfikatorach wytrenowanych na parach klas i zbieramy predykcje wskazujaca na konkretna klase
        przewidywane_etykiet = []

        for probka in X_test:
            class_votes = [0] * len(self.classes)

            for class_a, class_b, classif in self.classifiers:
                glos = classif.predict([probka])[0] #

                if glos >0:
                    class_votes[class_a] +=1
                else:
                    class_votes[class_b] +=1

            przewidywane_etykiety = np.argmax(class_votes) #wybiera klase z max,najwieksza iloscia glosow
            przewidywane_etykiet.append(przewidywane_etykiety) #etkieta dodawana do przewidywania etykiet

        return przewidywane_etykiet
    ######ONE VS REST##########
class one_vs_all(BaseEstimator):
    def __init__(self):
        self.classifiers = []

    def fit(self, X, y):
        self.classes = np.unique(y).flatten() #unique wyznacza unikalne etykiety, flatte splaszcza tablice
        for etykieta_klasy in self.classes:
            binarna_etykieta = np.where(y == etykieta_klasy, 1, 0) #jeśli etykieta należy do klasy 1 jeśli należy do każdej innej to 0
            classif = aSVM()
            classif.fit(X, binarna_etykieta)
            self.classifiers.append((etykieta_klasy, classif))
    def predict(self, X_test):
        przewidywane_etykiety = [] #przechowalnia przewidywan klas dla X_testa

        for probka in X_test:
            class_votes = [0] * len(self.classes) #wyniki głosów class_votes tablica wypelniona zerami

            for etykieta_klasy, classif in self.classifiers: #iteracja po każdej parze (etkieta klasy, klsyfikator)
                glos = classif.predict([probka])[0]  #wynik predykcji dla probka, [0] aby uzyskac pierwszy element z listy

                if glos > 0: #jeśli wartosc glosu jest większa od 0 to klasyfikator przewidzial ze probka nalezy do klasy
                    class_votes[etykieta_klasy] +=1
            przewidywanie_klass = np.argmax(class_votes) #wybiera indeks klasy z najwieksza liczba glosow, przewidywana klasa
            przewidywane_etykiety.append(przewidywanie_klass) #dodaje klase(etykiete) do listy przewidzianych etykiet
        return przewidywane_etykiety
