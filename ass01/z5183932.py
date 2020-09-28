import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

studentid = os.path.basename(sys.modules[__name__].__file__)


#################################################
import numpy as np
from collections import Counter, OrderedDict
from itertools import chain
from datetime import datetime
import re
#################################################


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))
    if other is not None:
        print(question, other)
    if output_df is not None:
        print(output_df.head(5).to_string())


def question_1(movies, credits):
    """
    :param movies: the path for the movie.csv file
    :param credits: the path for the credits.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    df_credit = pd.read_csv(credits)
    df_movie = pd.read_csv(movies, engine='python', error_bad_lines=False)
    df1 = pd.merge(left=df_credit, right=df_movie, on="id")
    #################################################

    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    to_drop = ['adult','belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path', 'status', 'tagline', 'video']
    df2 = df1.drop(to_drop, axis=1)
    #################################################

    log("QUESTION 2", output_df=df2, other=(len(df2.columns), sorted(df2.columns)))
    return df2


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    df3 = df2.set_index('id')
    #################################################

    log("QUESTION 3", output_df=df3, other=df3.index.name)
    return df3


def question_4(df3):
    """
    :param df3: the dataframe created in question 3
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    df4 = df3.drop(df3[df3.budget == 0].index)
    #################################################

    log("QUESTION 4", output_df=df4, other=(df4['budget'].min(), df4['budget'].max(), df4['budget'].mean()))
    return df4


def question_5(df4):
    """
    :param df4: the dataframe created in question 4
    :return: df5
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    impact = []
    for index, row in df4.iterrows():
        budget = row['budget']
        revenue = row['revenue']
        result = (revenue - budget)/budget
        impact.append(result)
    
    df4['success_impact'] = impact
    df5 = df4
    #################################################

    log("QUESTION 5", output_df=df5,
        other=(df5['success_impact'].min(), df5['success_impact'].max(), df5['success_impact'].mean()))
    return df5


def question_6(df5):
    """
    :param df5: the dataframe created in question 5
    :return: df6
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    df5['popularity'] = df5['popularity'] * 2/3
    df6 = df5
    #################################################

    log("QUESTION 6", output_df=df6, other=(df6['popularity'].min(), df6['popularity'].max(), df6['popularity'].mean()))
    return df6


def question_7(df6):
    """
    :param df6: the dataframe created in question 6
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    df7 = df6
    df7['popularity'] = df7['popularity'].astype('int16')
    #################################################

    log("QUESTION 7", output_df=df7, other=df7['popularity'].dtype)
    return df7


def question_8(df7):
    """
    :param df7: the dataframe created in question 7
    :return: df8
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    df8 = df7
    extracted = df8['cast'].apply(lambda st: (re.findall('\'character\':\s\'([^,]*)\'', st)))
    df8['cast'] = extracted.apply(sorted)
    df8['cast'] = df8['cast'].apply(lambda x: ', '.join(map(str, x)))
    #################################################

    log("QUESTION 8", output_df=df8, other=df8["cast"].head(10).values)
    return df8


def question_9(df8):
    """
    :param df9: the dataframe created in question 8
    :return: movies
            Data Type: List of strings (movie titles)
            Please read the assignment specs to know how to create the output
    """

    #################################################
    chs = df8['cast'].str.split(',')
    num = []
    for i in chs:
       num.append(len(i))
    df8['num'] = num
    df8 = df8.sort_values(by=['num'], ascending=False)
    result = df8.head(10)['title']
    movies = []
    for movie in result:
        movies.append(movie)
    #################################################

    log("QUESTION 9", output_df=None, other=movies)
    return movies


def question_10(df8):
    """
    :param df8: the dataframe created in question 8
    :return: df10
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    df10 = df8
    df10['release_date'] = df10['release_date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
    df10 = df10.sort_values(by=['release_date'], ascending=False)
    #################################################

    log("QUESTION 10", output_df=df10, other=df10["release_date"].head(5).to_string().replace("\n", " "))
    return df10


def question_11(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    #clean the dataset
    df10['genres'] = df10['genres'].apply(lambda x: re.findall('\'name\':\s\'([^,]*)\'', x))
    df10['genres'] = df10['genres'].apply(lambda x: ', '.join(map(str, x)))

    s = df10['genres']
    unival = Counter(map(str.strip, chain.from_iterable(s.str.split(','))))
    replace = ['Music', 'Western', 'Documentary', 'TV Movie']
    num = 0
    for name in replace:
        num = unival.pop(name) + num
    unival['other genres'] = num
    
    plt.rcParams['font.size'] = 8.0
    fig1, ax1 = plt.subplots()
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]

    ax1.pie(unival.values(),autopct='%.1f%%',  pctdistance=1.2, colors=colors)
    plt.legend(bbox_to_anchor=(1.10,1.025), loc="upper left", labels=unival.keys())
    plt.title('Genres')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
    #################################################

    plt.savefig("{}-Q11.png".format(studentid))


def question_12(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # clean the dataset
    df10['production_countries'] = df10['production_countries'].apply(lambda x: re.findall('\'name\':\s\'([^,]*)\'', x))
    df10['production_countries'] = df10['production_countries'].apply(lambda x: ', '.join(map(str, x)))

    s = df10['production_countries']

    unival = Counter(map(str.strip, chain.from_iterable(s.str.split(','))))
    unival = OrderedDict(sorted(unival.items()))

    plt.rcParams['figure.figsize'] =(15,8)
    fig, ax = plt.subplots()
    plt.bar(unival.keys(), unival.values(), width=0.8, linewidth=0.1)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(np.arange(0,500,step=20))
    plt.subplots_adjust(left=0.1, bottom=0.32, right=0.75, hspace = 0.2)
    
    for i, v in enumerate(unival.values()):
        ax.text(i-0.3, v+5, str(v), fontsize=8)
    ax.set_title('production country')
    #################################################

    plt.savefig("{}-Q12.png".format(studentid))


def question_13(df10):
    """
    :param df10: the dataframe created in question 10
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    df10 = df10[['original_language', 'vote_average', 'success_impact']]
    groups = df10.groupby('original_language')

    plt.rcParams['figure.figsize'] =(15,8)
    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.vote_average, group.success_impact, marker='o', linestyle='', ms=0.5, label=name)
        ax.legend(bbox_to_anchor=(1,1.025), loc="upper left", fontsize='small')
    plt.x=([-1, 5500])
    ax.set_ylabel('success_impact')
    ax.set_xlabel('vote_average')
    ax.set_title('vote_average vs success_impact')
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.8, hspace = 0.2)
    #################################################

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("movies.csv", "credits.csv")
    df2 = question_2(df1)
    df3 = question_3(df2)
    df4 = question_4(df3)
    df5 = question_5(df4)
    df6 = question_6(df5)
    df7 = question_7(df6)
    df8 = question_8(df7)
    movies = question_9(df8)
    df10 = question_10(df8)
    question_11(df10)
    question_12(df10)
    question_13(df10)