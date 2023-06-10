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
        #self.classifier = classifier
        self.classes = []
    def fit(self, X, y):

        self.classes = np.unique(y)#.flatten()

        for class_a, class_b in combinations(self.classes, 2): #wybiera wszystkie możliwe pary
            classif = aSVM()
            mask = (y == class_a) | (y == class_b) #True jeśli należą do jednej z dwóch klas ('a' lub 'b') jak nie to False
            #szablon_X = (X == class_a) | (X == class_b)
            X_pair = X[mask] # tylko te elementy ze zbioru które są true, podzbiór danych treningowych dla danej pary klas
            #print('X[szablon] OVO: ', para_X) #print daje wartości danych treningowych [1,3222 -0,541424 2,55555 1,2234] itd
            y_pair = y[mask] #etykiety ktore odpowiadaja wartosci TRUE, podzbior etykiet dla danej pary klas
            #print('y[szablon] OVO: ', para_y) #print wyswietla arraya podzbioru etykiet dla kazdej pary klas [1 2 1 2 2 2 1 1] [0 0 00 011] itd.

            y_binary = np.where(y_pair == class_a, 1, -1)
            classif.fit(X_pair, y_binary)

            self.classifiers.append((class_a, class_b, classif)) #dodaje do arraya classifiers
            #self.classifiers = np.array(self.classifiers).flatten('C')
        return self
    def predict(self, X_test): #glosowanie wiekszosciowe ta klasa co ma najwiecej glosow wygrywa, iteracja po all binarnych klasyfikatorach wytrenowanych na parach klas i zbieramy predykcje wskazujaca na konkretna klase
        przewidywane_etykiet = []

        for probka in X_test:
            class_votes = [0] * len(self.classes)

            for class_a, class_b, classif in self.classifiers:
                glos = classif.predict([probka])[0]

                class_a_idx = np.where(self.classes == class_a)[0][0] #porównanie elementow w tablicy self.classes z class_a, zwraca ablice indeksów w ktorych wartosci sa rowne class_a
                #pobranie wszystich indeksow dla klasy_a
                class_b_idx = np.where(self.classes == class_b)[0][0]
               # if glos == class_a:
                #    class_votes[class_a] +=1
                #elif glos ==class_b:
                #    class_votes[class_b] +=1
                if glos > -1 :#-1:
                    class_votes[class_a_idx] +=1
                else:
                    class_votes[class_b_idx] +=1
            #print('class_votes ovo a: ',class_votes[class_a],'class_votes ovo b: ', class_votes[class_b])
            #print('class votes: ', class_votes)
            przewidywane_etykiety = np.argmax(class_votes) #wybiera klase z max,najwieksza iloscia glosow
            przewidywane_etykiet.append(self.classes[przewidywane_etykiety]) #etkieta dodawana do przewidywania etykiet
            #print('przewidywanie etykiet ovo: ', przewidywane_etykiety)
            #print('przewidywane etykiet ovo:', przewidywane_etykiet)
        print('przewidywane etykiet OVO array: ', np.array(przewidywane_etykiet))
        return przewidywane_etykiet
    ######ONE VS REST##########
class one_vs_all(BaseEstimator):
    def __init__(self):
        self.classifiers = []

    def fit(self, X, y):
        self.classes = np.unique(y).flatten() #unique wyznacza unikalne etykiety, flatte splaszcza tablice
        self.ilosc_klas = len(self.classes) #oblicza liczbe klas na podstawie unikalnych etykiet
        #self.classifiers = [] #pusta tablica na pary (etykieta klasy, klasyfikator), wielkosc macierzy 2 na ilosc poszczegolnych klas
        for etykieta_klasy in self.classes: #iteracja po unikalnych etykietach klasy
            binarna_etykieta = np.where(y == etykieta_klasy, 1, -1) #jeśli etykieta należy do klasy 1 jeśli należy do każdej innej to -1!!!!
            classif = aSVM() #użycie aSVMa
            classif.fit(X, binarna_etykieta) #trenowanie klasyfikatora na podstawie danych X i etykiet binarnych
            #po wykonaniu fita klasyfikator gotowy jesy do przewidywania etykiet
            self.classifiers.append((etykieta_klasy, classif)) #dodaje parę (etkieta, klasyfikator) do listy  classifiers, append(rekord dodawany na koncu listy)

    def predict(self, X_test):

        ilosc_probek = X_test.shape[0]  #ilosc probek testowych
        macierz_wsparc = np.zeros((ilosc_probek, self.ilosc_klas)) #macierz o wymiarach(ilosc probek na ilosc klas) wypełniona zerami, celem jest przechowyanie wsparcia dla klas
        for i in range(self.ilosc_klas): #iteracja po klasach
            etykieta_klasy, clf = self.classifiers[i] #przypisanie wartosci z listy classifiers[i](w ktorej są pary etkieta klasy, klasyfikator) do zmiennych etykieta_klasy i clf
            wsparcie = clf.predict(X_test) #zwraca wartość wsparcia dla danych testowych
            #print('Wsparcie predict OVA: ', wsparcie) #printuje liste 1 i -1
            macierz_wsparc[:,i] = wsparcie #przypisanie wartosci wsparcia dla danej klasy do odpowiedniej kolumny macierzy wsparcia
            """
            macierz_wsparcia przechowuje wartosci wsparcia klas dla wszystkich probek testowych, iteracja przez wszystkie klasy;
            dla kazdej klasy 'wsparcie' bedzie zawierac wartosci wsparcia dla probek testowych dla konkretnej klasy
            """
            #print('Macierz wsparć predict OVA: ', macierz_wsparc[:,i]) #printuje liste 1 i -1
        przewidywane_etykiety  = np.argmax(macierz_wsparc, axis=1)#argmax zwraca index elementu o najwiekszej wartosci a tym wypadku etykiete
        # o najwiekszym wsparciu, axis=1 - przeszukanie w każdym wierszu macierzy,
        # przywidywanie_etykiet to tablica z indekasami kolumn z najwiekszymi wsparciami dla kazdego wiersza macierzy macierz_wsparc
        print('PRZEWIDYWANIE ETYKIET PREDICT OVA: ', przewidywane_etykiety) #print daje liste etykiet[0 1 2.....]
        return przewidywane_etykiety #zwraca przewidywanie_etykiet