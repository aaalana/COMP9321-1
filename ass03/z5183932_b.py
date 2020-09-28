import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier




'''
first try:
cast -> top four actors' id
crew -> director's id(""id"": 15344, ""job"": ""Director"")
        writer
        producer
genres considered as multi label class
keywords multi label
original_language
production_companies multi label
production_countries multi label
release_date -> year
runtime
number of spoken_language
'''

def update_df(df, attributes, info):
    for attribute in attributes:
        all_scores = []
        #calculate mean scores to replace missing value
        mean = (np.array(list(info[attribute].values())).mean())
        for values in df[attribute]:
            scores = 0
            count = 0
            if values != '0':
                values = values.split(',')
                #print(values)
                for value in values:
                    try:
                        scores += info[attribute][value]
                        count += 1
                    except:
                        pass
                if scores == 0:
                    scores == mean
                else:
                    scores = scores/count
            else:
                scores = mean
            all_scores.append(scores)
        se = pd.Series(all_scores)
        df[attribute] = se
    
    return df
            


#return a dictionary of atttibutes need to be preproccessed
#{'writers':{'id': xxx}, {}..
#'director': {'id': xxx}, {}..
#'actors':
#'genres':
# }
#xxx is the mean score of movies related to the attricute
def preprocess(df, attributes, unique_set, column):
    info = {}
    #print(df[df['keywords'].str.contains('1463')])
    for attribute in attributes:
        info[attribute] = {}
        for value in unique_set[attribute]:
            tmp = pd.DataFrame(df[df[attribute].str.contains(value)])
            rating = tmp[column].mean()
            info[attribute][value] = rating
    return info

def get_unique(df, attributes):
    unique_s = {}
    for attribute in attributes:
        unique_l = []
        if "actor" in attribute:
            for value in df[attribute]:
                if value not in unique_l and value != 0:
                    unique_l.append(value)
        else:
            for values in df[attribute]:
                if values != str(0):
                    values = values.split(",")
                    for value in values:
                        if value not in unique_l:
                            unique_l.append(value)
        unique_s[attribute] = unique_l
    return unique_s
                
#extract id from json format data, return narray type of data
def ext_id(column):
    result = []
    for movie in column:
        movie = json.loads(movie)
        items = []
        for item in movie:
            try:
                item = str(item['id'])
            except:
                item = str(item['iso_3166_1'])
            items.append(item)

        if items == []:
            items = str(0)
        else:
            items = ",".join(items)
        result.append(items)
    return result


#accepct narray type of data and convert it into label encoded data
def label_encode(data):
    #do the label encoding for catorial feature
    le = LabelEncoder()
    se = data.apply(lambda col: str(col))
    se = le.fit_transform(se)
    return se

def clean_x(df):
    #extract top 4 actors from cast
    actor_1 = []
    actor_2 = []
    actor_3 = []
    actor_4 = []
    actors = [actor_1,actor_2,actor_3,actor_4]
    actors_name = ['actor_1','actor_2','actor_3','actor_4']
    for movie in df.cast:
        movie = json.loads(movie)
        i = 0
        for i in range(4):
            try:
                actor = list(movie)[i]
                actor = str(actor['id'])
            except:
                actor = str(0)
            if i == 0:
                actor_1.append(actor)
            elif i == 1:
                actor_2.append(actor)
            elif i == 2:
                actor_3.append(actor)
            elif i == 3:
                actor_4.append(actor)
            i += 1

    i = 0
    for actor in actors:
        se = pd.Series(actor)
        df[actors_name[i]] = se
        i += 1
    
    #extract director, writer, producer from crew
    all_writers = []
    all_directors = []
    all_producers = []

    jobs = [all_directors, all_writers, all_producers]
    jobs_name = ['all_directors', 'all_writers', 'all_producers']

    for movie in df['crew']:
        movie = json.loads(movie)
        writers = []
        directors = []
        producers = []
        for crew in movie:
            if crew['job'] == "Writer":
                crew = str(crew['id'])
                writers.append(crew)
            elif crew['job'] == "Director":
                crew = str(crew['id'])
                directors.append(crew)
            elif crew['job'] == 'Producer':
                crew = str(crew['id'])
                producers.append(crew)
        writers = ','.join(writers)
        producers = ','.join(producers)
        directors = ','.join(directors)

        all_directors.append(directors)
        all_producers.append(producers)
        all_writers.append(writers)

    i = 0
    for job in jobs:
        se = pd.Series(job)
        df[jobs_name[i]] = se
        i += 1
    #only keep the id of genres
    catorial_feature = ['genres', 'keywords', 'production_companies', 'production_countries']
    for feature in catorial_feature:
        se = ext_id(df[feature])
        df[feature] = se

    #split the release date to year and month
    years = []
    months = []
    for date in df['release_date']:
        year = int(date[:4])
        month = int(date[5:7])
        years.append(year)
        months.append(month)
    se1 = pd.Series(years)
    se2 = pd.Series(months)
    df['year'] = se1
    df['month'] = se2

    #around runtime to hours
    df['runtime'] = round(df['runtime']/60, 1)

    df = df.fillna(0)
    
    return df

def load_x(df):
    tmp = df.drop(['revenue', 'rating'], axis=1)
    result = clean_x(tmp)
    return result

def load_y(df):
    result = df['rating']
    df.fillna(0, inplace=True)
    return result

if __name__ == '__main__':
    
    jobs_name = ['all_directors', "all_writers", "all_producers"]
    catorial_feature = ['genres', 'keywords', "production_countries"]
    actors= ['actor_1', 'actor_2', 'actor_3', 'actor_4']
    others = ['year', 'budget']
    
    train = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

    #regression
    train_x = clean_x(train)
    train_y = train['revenue']
    test_x = clean_x(test)
    test_y = test['revenue']

    #method 1
    #get the a dictionary of unique ids in  catorial_feature and jobs_name
    unique_set = get_unique(train_x, catorial_feature+jobs_name+actors)

    #get a dictionary of id and mean score of movie
    info = preprocess(train_x, catorial_feature+jobs_name+actors, unique_set, 'revenue')

    #replace certain cloumn in dataframe
    train_x = update_df(train_x, catorial_feature+jobs_name+actors, info)

    unique_set = get_unique(test_x, catorial_feature+jobs_name+actors)
    info = preprocess(test_x, catorial_feature+jobs_name+actors, unique_set, 'revenue')
    test_x = update_df(test_x, catorial_feature+jobs_name+actors, info)

    test_x = test_x[jobs_name+actors+catorial_feature+others]
    train_x = train_x[jobs_name+actors+catorial_feature+others]

    reg_model = LinearRegression()
    # reg_model = XGBRegressor(colsample_bytree= 0.6, gamma= 0.7, max_depth= 4, min_child_weight= 5,
    #                      subsample = 0.8, objective='reg:squarederror')
    reg_model.fit(train_x, train_y)

    y_pred = reg_model.predict(test_x)

    # The mean squared error
    # for i in range(len(y_pred)):
    #     print("Expect value: {}".format(test_y[i]) + '\t' + 'Real value: {}'.format(y_pred[i]))
    print("Mean squared error: %.2f"% mean_squared_error(test_y, y_pred))
    print(r2_score(test_y,y_pred))
    print("Correlation: {}".format(pearsonr(test_y, y_pred)))
    print("Training set score: {:.2f}".format(reg_model.score(train_x, train_y)))

    features = train_x
    targets = train_y
    cv_scores = cross_val_score(reg_model, features, targets,cv=10, scoring='r2')

    #outputs the scores
    print('Cross Validation scores: {}'.format(cv_scores))
    print("\nAverage 10-Fold CV Score: {}".format(np.mean(cv_scores)))
    print(pd.DataFrame({'features':train_x.columns,'coeficients':reg_model.coef_[0]}))

   

    #filtering the dataframe to include just the IV's (features) and another with just the DV's (target)
    #creating pairwise correlation of columns using Pearson's R,
    corr = features.corr(method='pearson') 

    #plotting the correlation coefficients on a seasborn heatmap matrix
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, xticklabels=corr.columns, 
                yticklabels=corr.columns, ax=ax, linewidths=.5, 
                vmin = -1, vmax=1, center=0)

    plt.title('Correlation HeatMap')
    plt.show()


    




    #classification

    # jobs_name = ['all_directors', 'all_writers']
    # catorial_feature = []
    # actors= ['actor_1', 'actor_2']
    # others = []

    # train = pd.read_csv(sys.argv[1])
    # test = pd.read_csv(sys.argv[2])

    # train_x = clean_x(train)
    # train_y = train['rating']
    # test_x = clean_x(test)
    # test_y = test['rating']

    # #get the a dictionary of unique ids in  catorial_feature and jobs_name
    # unique_set = get_unique(train_x, catorial_feature+jobs_name+actors)

    # #get a dictionary of id and mean score of movie
    # info = preprocess(train_x, catorial_feature+jobs_name+actors, unique_set, 'rating')

    # #replace certain cloumn in dataframe
    # train_x = update_df(train_x, catorial_feature+jobs_name+actors, info)

    # unique_set = get_unique(test_x, catorial_feature+jobs_name+actors)
    # info = preprocess(test_x, catorial_feature+jobs_name+actors, unique_set, 'rating')
    # test_x = update_df(test_x, catorial_feature+jobs_name+actors, info)

    # train_x = train_x[catorial_feature+jobs_name+actors + others]
    # test_x = test_x[catorial_feature+jobs_name+actors + others]

    # knn = KNeighborsClassifier()
    # model = knn.fit(train_x, train_y)
    # predictions = knn.predict(test_x)
    # print(predictions)
    # print("confusion_matrix:\n", confusion_matrix(test_y, predictions))
    # print("precision:\t", precision_score(test_y, predictions, average=None))
    # print("recall:\t", recall_score(test_y, predictions, average=None))
    # print("accuracy:\t", accuracy_score(test_y, predictions))

    # cda =  LogisticRegression()
    # model = cda.fit(train_x, train_y)
    # predictions = cda.predict(test_x)
    # print(predictions)
    # print("confusion_matrix:\n", confusion_matrix(test_y, predictions))
    # print("precision:\t", precision_score(test_y, predictions, average=None))
    # print("recall:\t", recall_score(test_y, predictions, average=None))
    # print("accuracy:\t", accuracy_score(test_y, predictions))