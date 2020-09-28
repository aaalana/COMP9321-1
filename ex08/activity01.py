import pandas as pd

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

def iris_load(df, percentage):
    #shuffle the ordered dataframe
    df = shuffle(df)

    iris_x = df.drop('species', axis=1).values
    iris_y = df['species'].values

    split_point = int(len(iris_x) * percentage)
    iris_X_train = iris_x[:split_point]
    iris_Y_train = iris_y[:split_point]
    iris_X_test = iris_x[split_point:]
    iris_Y_test = iris_y[split_point:]

    return iris_X_train, iris_Y_train, iris_X_test, iris_Y_test
    
    

if __name__ == "__main__":
    df = pd.read_csv('iris.csv')
    
    #split the data into train and test sets
    iris_X_train, iris_Y_train, iris_X_test, iris_Y_test = iris_load(df, 0.7)

    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_Y_train)

    predictions = knn.predict(iris_X_test)

    print("confusion_matrix:\n", confusion_matrix(iris_Y_test, predictions))
    #precision = number of true positive/true positive + false positive = 14/15
    print("precision:\t", precision_score(iris_Y_test, predictions, average=None))
    #recall ratio = number of tp/tp + fn = 17/18
    print("recall:\t\t", recall_score(iris_Y_test, predictions, average=None))
    #accuracy = 44/45
    print("accuracy:\t", accuracy_score(iris_Y_test, predictions))