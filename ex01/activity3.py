import pandas as pd
import json
from pymongo import MongoClient


def write_in_mongodb(dataframe, mongo_host, mongo_port, db_name, collection):
    """
    :param dataframe: 
    :param mongo_host: Mongodb server address 
    :param mongo_port: Mongodb server port number
    :param db_name: The name of the database
    :param collection: the name of the collection inside the database
    """
    client = MongoClient(host=mongo_host, port=mongo_port)
    db = client[db_name]
    c = db[collection]
    # You can only store documents in mongodb;
    # so you need to convert rows inside the dataframe into a list of json objects
    records = json.loads(dataframe.T.to_json()).values()
    c.insert_many(records)

def read_from_mongodb(mongo_host, mongo_port, db_name, collection):
    """
    :param mongo_host: Mongodb server address 
    :param mongo_port: Mongodb server port number
    :param db_name: The name of the database
    :param collection: the name of the collection inside the database
    :return: A dataframe which contains all documents inside the collection
    """
    client = MongoClient(host=mongo_host, port=mongo_port)
    db = client[db_name]
    c = db[collection]
    return pd.DataFrame(list(c.find()))

def print_dataframe(dataframe):
    print(','.join(column for column in dataframe))

    for index, row in dataframe.iterrows():
        print(','.join(str(row[column] for column in dataframe)))

if __name__=='__main__':
    db_name = 'comp9321'
    mongo_port = 27017
    mongo_host = 'localhost'
    dataframe = pd.read_csv("Demographic Statistics.csv")

    collection = 'Demographic_Statistics'
    write_in_mongodb(dataframe, mongo_host, mongo_port, db_name, collection)

    df = read_from_mongodb(mongo_host, mongo_port, db_name, collection)
    print_dataframe(df)
    

