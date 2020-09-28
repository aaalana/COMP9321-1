import pandas as pd
import sqlite3
from pandas.io import sql
import activity1 as a1

if __name__=="__main__":
    csv_file="Demographic Statistics.csv"
    #read the dataframe
    dataframe = pd.read_csv(csv_file)
    database_file = "Demographic_Statistics.db"
    table_name = "Demographic_Statistics"

    #store the dataframe to sqlite
    #DataFrame.to_sql(name, con, flavor=None, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None)[source]
    cnn = sqlite3.connect(database_file)
    sql.to_sql(dataframe, name=table_name, con=cnn, if_exists='append')


    #read data from sqlite
    queried = sql.read_sql('select * from ' + table_name, cnn)
    
    #write out
    dataframe.to_csv('Demographic Statistics_NEW_SQLITE.csv')
