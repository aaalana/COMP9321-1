import pandas as pd
import q02
import q01
def new_df():
    df = q02.new_df()
    new_df = df.set_index('id')
    return new_df

if __name__=='__main__':
    df = new_df()
    # print(df)