import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime
from pandas.io import sql
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
import urllib.request
from flask import request
from flask import Flask
from flask_restplus import Resource, Api
from flask_restplus import fields, inputs, reqparse

def fetchlocal(file_name):
    with open(file_name, 'r') as f:
        distros_dict = json.load(f)
    
    return distros_dict

def fetch(filename):
    url = "http://api.worldbank.org/v2/countries/all/indicators/{}\
        ?date=2012:2017&format=json&per_page=1000".format(filename)
    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())
    return data

def json_to_df(json_file, filename):
    cnn = sqlite3.connect("z5183932.db")
    c = cnn.cursor()
    json_data = json_file[1]
    df = pd.DataFrame()
    max = 0
    #check if index exist in database
    try:
        c.execute("select * from collections where [indicator.id] = \"{}\"".format(filename))
        if c.fetchall():
            return df
    except:
        pass
    #select the max collection_id
    try:
        c.execute('''select max(collection_id) from collections''')
        max = c.fetchone()[0]
    except:
        pass
    #add the collection_id
    for d in json_data:
        d['collection_id'] = max + 1
    df = pd.json_normalize(json_data)
    #clear entires with nan
    df.dropna(inplace=True)
    return df

def store_df(df):
    cnn = sqlite3.connect("z5183932.db")
    sql.to_sql(df, name="collections", con=cnn, if_exists="append")
    cnn.close()

if __name__=="__main__":
    # url = "1.1.PSev.Poor4uds.json"
    # filename = "1.1.PSev.Poor4uds"
    # data = fetchlocal(url)
    # df = json_to_df(data, filename)
    # if df.empty:
    #     print('error')
    # else:
    #     store_df(df)

    #print(df)
    data = fetch("1.0.PGap.2.5usd")
    print(data)
