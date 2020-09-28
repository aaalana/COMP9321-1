import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import re
import q10

def clean(df):
    df['genres'] = df['genres'].apply(lambda x: re.findall('\'name\':\s\'([^,]*)\'', x))
    df['genres'] = df['genres'].apply(lambda x: ', '.join(map(str, x)))

    return df

if __name__=='__main__':
    df = q10.new_df()
    df = clean(df)
    s = df['genres']
    unival = Counter(map(str.strip, chain.from_iterable(s.str.split(','))))
    # print(df['genres'].to_string())
    replace = ['Music', 'Western', 'Documentary', 'TV Movie']
    num = 0
    for name in replace:
        num = unival.pop(name) + num
    unival['other genres'] = num
    
    fig1, ax1 = plt.subplots()
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]

    ax1.pie(unival.values(), labels=unival.keys(), autopct='%1.0f%%',  pctdistance=1.2, labeldistance=None, colors=colors)
    plt.legend(bbox_to_anchor=(1.10,1.025), loc="upper left")
    plt.title('Genres')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
    plt.show()