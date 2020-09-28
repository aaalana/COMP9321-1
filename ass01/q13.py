import pandas as pd
import q10
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    df = q10.new_df()
    df = df[['original_language', 'vote_average', 'success_impact']]
    groups = df.groupby('original_language')

    plt.rcParams['figure.figsize'] =(15,8)
    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.success_impact, group.vote_average, marker='o', linestyle='', ms=1, label=name)
        ax.legend(bbox_to_anchor=(1,1.025), loc="upper left", fontsize='small')
    plt.x=([-1, 5500])
    ax.set_xlabel('success_impact')
    ax.set_ylabel('vote_average')
    ax.set_title('vote_average vs success_impact')
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.8, hspace = 0.2)
    plt.show()