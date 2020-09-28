import pandas as pd
import re
import numpy as np
import q10
from collections import Counter, OrderedDict
from itertools import chain
import matplotlib.pyplot as plt


def clean(df):
    df['production_countries'] = df['production_countries'].apply(lambda x: re.findall('\'name\':\s\'([^,]*)\'', x))
    df['production_countries'] = df['production_countries'].apply(lambda x: ', '.join(map(str, x)))
    return df

if __name__=='__main__':
    df = q10.new_df()
    df = clean(df)
    s = df['production_countries']

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
    plt.show()
