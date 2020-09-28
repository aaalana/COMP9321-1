import pandas as pd
import json
import q07
import re

def new_df():
    df = q07.new_df()
    # extracted = df.cast.str.extractall(r'\'character\':\s\'([^,]*)\'')
    extracted = df['cast'].apply(lambda st: (re.findall('\'character\':\s\'([^,]*)\'', st)))
    df['cast'] = extracted.apply(sorted)
    df['cast'] = df['cast'].apply(lambda x: ', '.join(map(str, x)))
    return df

if __name__=='__main__':
    df = new_df()
    #print(df.dtypes)
    print(df)