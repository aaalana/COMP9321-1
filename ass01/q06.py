import pandas as pd
import q05

def new_df():
    df = q05.new_df()
    df['popularity'] = df['popularity'] * 2/3
    return df


if __name__=='__main__':
    df = new_df()