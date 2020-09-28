import pandas as pd
import q03

def new_df():
    df = q03.new_df()
    new_df = df.drop(df[df.budget == 0].index)
    return new_df

if __name__=='__main__':
    df = new_def()
    print(df)