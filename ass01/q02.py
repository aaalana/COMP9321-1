import pandas as pd
import q01

def new_df():
    df = q01.new_df()
    to_drop = ['adult','belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path', 'status', 'tagline', 'video']
    new_df = df.drop(to_drop, axis=1)
    return new_df

if __name__=='__main__':
    df = new_df()
    q01.printcolumn(df)

