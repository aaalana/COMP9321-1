import pandas as pd
import q08

def get_movie():
    df = q08.new_df()
    chs = df['cast'].str.split(',')
    num = []
    for i in chs:
        print(type(i))
        num.append(len(i))
    df['num'] = num
    df = df.sort_values(by=['num'], ascending=False)
    result = df.head(10)['title']
    movies = []
    for movie in result:
        movies.append(movie)
    return movies

if __name__=='__main__':
    movies = get_movie()