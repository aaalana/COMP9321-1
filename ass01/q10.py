import pandas as pd
import q08
from datetime import datetime

def new_df():
    df = q08.new_df()
    df['release_date'] = df['release_date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
    df = df.sort_values(by=['release_date'], ascending=False)
    return df

if __name__=='__main__':
    df = new_df()
    # print(df['release_date'].head().to_string())
