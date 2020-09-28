import pandas as pd
import q06

def new_df():
    df = q06.new_df()
    df['popularity'] = df['popularity'].astype('int16')
    return df

if __name__=='__main__':
    df = new_df()
    print(df.dtypes)