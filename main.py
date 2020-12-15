# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sklearn as sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def readCSV (StringName) :
    f = pd.read_csv(StringName)
    return f

def IDDropper (dataFrame):
    dataFrame.drop('Loan_ID',axis=1,inplace=True)
    return dataFrame

def Concater (dataFrame1, dataFrame2) :
    X = pd.concat([dataFrame1,dataFrame2], axis=1)
    return X


def YTransformToBin(dataFrame):
    output_vals = {'Y':1, 'N':0}
    output = dataFrame['Loan_Status']
    dataFrame.drop('Loan_Status',axis=1, inplace=True)
    output=output.map(output_vals)
    return output

def List_CatTransformer(dataFrame):
    le = LabelEncoder()
    for col in dataFrame :
        dataFrame[col]= le.fit_transform(dataFrame[col])
    return dataFrame

def list_splitter (file):
    ListeCategorique =[]
    ListeNumerique = []

    for col,ty_pe in enumerate(file.dtypes):
        if ty_pe==object :
            ListeCategorique.append(file.iloc[:,col])
        else :
            ListeNumerique.append(file.iloc[:,col])

    return (ListeNumerique,ListeCategorique)

def ListToDataFrame(List):
    List = pd.DataFrame(List).transpose()
    return List

def FillNullValuesForCat (dataFrame):
    dataFrame=dataFrame.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return dataFrame

def FillNullValuesForNum (dataFrame):
    dataFrame.fillna(method="bfill",inplace=True)
    return dataFrame

def TrainTestSplitter (X, outs):
    Spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train, test in Spliter.split(X, outs):
        X_train, X_test = X.iloc[train], X.iloc[test]
        Y_train, Y_test = outs.iloc[train], outs.iloc[test]
    return (X_train,X_test,Y_train,Y_test)

def accuracy (expected, predicted, model_name, returnerPrinter ):
    acc = accuracy_score(expected,predicted)
    if returnerPrinter :
        return acc
    else :
        print ('La précision du modèle ',model_name,f' est de : {acc}')

def Evaluation (modeles, X_train, Y_train, X_test, Y_test) :
    List =[]

    for tag, model in modeles.items() :
        print (tag, " : ")
        model.fit (X_train,Y_train)
        accuracy(Y_test, model.predict(X_test), tag, False)
        List.append([tag, accuracy(Y_test, model.predict(X_test), tag, True)])
    df = pd.DataFrame(List, columns=['x', 'y'])
    plt.plot('x', 'y', data=df)
    plt.xticks(rotation='vertical')
    plt.show()





def Cross_validation (modeles, X_train, Y_train) :
    for tag, model in modeles.items() :
        classfier =model
        scores = cross_val_score(classfier, X_train, Y_train, cv=5)
        print ("Cross-validation score for " ,tag, " : ",scores )


# lecture du fichier csv
file = readCSV("train_u6lujuX_CVtuZ9i.csv")
#config de l'affichage
pd.set_option('display.max_rows',file.shape[0]+1)

print (file)

#affichage 10 lignes max
pd.set_option('display.max_rows',10)
print (file)


file.info()
print(file.isnull().sum().sort_values(ascending=False))

#print(file.describe())

#print(file.describe(include='0'))



#split du dataset en set des variables qualitatives et quantitatives
l,l2 = list_splitter(file)

#transformer les listes en DataFrames
l=ListToDataFrame(l)
l2=ListToDataFrame(l2)


#remplacement des valeurs manquantes
l=FillNullValuesForNum(l)
l2=FillNullValuesForCat(l2)


print (l)
print("---------------------")
print(l2)
#print(l2.isnull().sum().sort_values(ascending=False))
#print(l.isnull().sum().sort_values(ascending=False))


#transformation de la sortie
outs = YTransformToBin(l2)
print(outs)

print (l2)


#encodage des variables qualitatives
l2=List_CatTransformer(l2)
print(l2)

#suppression de l'id
l2=IDDropper(l2)
print(l2)


#reconcaténation aprés traitement
X= Concater(l,l2)

#affichage des compteurs de sortie
print(outs.value_counts())



# ******************************************************************* Analyse exploratoire *****************************************************************


# graphes des sorties
plt.figure(figsize=(8,6))
sns.countplot(outs)
plt.show()


ClnDataSet = pd.concat([l,l2,outs], axis=1)
print(ClnDataSet)



grid = sns.FacetGrid(ClnDataSet,col='Loan_Status', size = 3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History')
plt.show()




grid = sns.FacetGrid(ClnDataSet,col='Loan_Status', size = 3.2, aspect=1.6)
grid.map(sns.countplot, 'Gender')
plt.show()



grid = sns.FacetGrid(ClnDataSet,col='Loan_Status', size = 3.2, aspect=1.6)
grid.map(sns.countplot, 'Married')
plt.show()

grid = sns.FacetGrid(ClnDataSet,col='Loan_Status', size = 3.2, aspect=1.6)
grid.map(sns.countplot, 'Self_Employed')
plt.show()

grid = sns.FacetGrid(ClnDataSet,col='Loan_Status', size = 3.2, aspect=1.6)
grid.map(sns.countplot, 'Education')
plt.show()

grid = sns.FacetGrid(ClnDataSet,col='Loan_Status', size = 3.2, aspect=1.6)
grid.map(sns.countplot, 'Property_Area')
plt.show()

plt.scatter(ClnDataSet['LoanAmount'], ClnDataSet['Loan_Status'])
plt.show()

plt.scatter(ClnDataSet['Loan_Amount_Term'], ClnDataSet['Loan_Status'])
plt.show()

plt.scatter(ClnDataSet['Dependents'], ClnDataSet['Loan_Status'])
plt.show()

plt.scatter(ClnDataSet['ApplicantIncome'], ClnDataSet['Loan_Status'])
plt.show()

plt.scatter(ClnDataSet['CoapplicantIncome'], ClnDataSet['Loan_Status'])
plt.show()



# ******************************************************************* tests & réalisation *****************************************************************


X_train, X_test, Y_train, Y_test = TrainTestSplitter(X,outs)

print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)

# models à tester régression logistique, KNN, Arbres de décision
# for KNN weight functio, = uniform & algorithm = auto algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’

models = {
    'REGLOG' : LogisticRegression (random_state=42),
    'KNN_3NN' : KNeighborsClassifier(n_neighbors=3),
    'KNN_4NN' : KNeighborsClassifier(n_neighbors=4),
    'KNN_5NN' : KNeighborsClassifier(n_neighbors=5),
    'KNN_6NN' : KNeighborsClassifier(n_neighbors=6),
    'KNN_7NN' : KNeighborsClassifier(n_neighbors=7),
    'KNN_8NN' : KNeighborsClassifier(n_neighbors=8),
    'KNN_9NN' : KNeighborsClassifier(n_neighbors=9),

    'DT_D2' : DecisionTreeClassifier(max_depth=2, random_state=42),
    'DT_D3' : DecisionTreeClassifier(max_depth=3, random_state=42),
    'DT_D4' : DecisionTreeClassifier(max_depth=4, random_state=42),
    'DT_D5' : DecisionTreeClassifier(max_depth=5, random_state=42),
}


Cross_validation(models,X_train,Y_train)
Evaluation(models,X_train,Y_train,X_test,Y_test)



X_Optimized = X[['Credit_History','Gender','ApplicantIncome','CoapplicantIncome','Self_Employed','Property_Area']]
X_train, X_test, Y_train, Y_test = TrainTestSplitter(X_Optimized,outs)
Cross_validation(models,X_train,Y_train)
Evaluation(models,X_train,Y_train,X_test,Y_test)

X_Optimized = X[['Credit_History','Gender','ApplicantIncome','CoapplicantIncome']]
X_train, X_test, Y_train, Y_test = TrainTestSplitter(X_Optimized,outs)


Cross_validation(models,X_train,Y_train)
Evaluation(models,X_train,Y_train,X_test,Y_test)

#Choix de la régression comme modèle
#Classifier = LogisticRegression()
#Classifier.fit(X_Optimized, outs)

#Enregistrelent du modèle
#pickle.dump(Classifier, open('model.pkl', 'wb'))

