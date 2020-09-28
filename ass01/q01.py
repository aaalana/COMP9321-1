import pandas as pd

def new_df():
    csv_credit = "credit.csv"
    csv_movie = "movie.csv"
    df_credit = pd.read_csv(csv_credit)
    df_movie = pd.read_csv(csv_movie, engine='python', error_bad_lines=False)
    df = pd.merge(left=df_credit, right=df_movie, on="id")
    return df

def printcolumn(df):
    print(','.join([column for column in df]))

def printdataframe(df):
    print(','.join([column for column in df]))

    for index, row in df.iterrows():
        print(','.join([str(row[column]) for column in df]))

if __name__=='__main__':
    df = new_df()
    printcolumn(df)
