import json
import sqlite3
import requests
import json
import pandas as pd
import urllib.request
from pandas.io import sql
from datetime import datetime
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import request
from flask import Flask
from flask import jsonify
from flask import make_response
from flask import Response
from flask_restplus import Resource, Api
from flask_restplus import fields, inputs, reqparse

app = Flask(__name__)
api = Api(app, default="Collections", title="GDP Collections Dataset",
        description="This is data on the GDP of all countries from 2012 to 2017")

app.config['JSON_SORT_KEYS'] = False

post_parser = reqparse.RequestParser()
post_parser.add_argument('indicator_id')

get_parser = reqparse.RequestParser()
get_parser.add_argument('order_by', help="please choose columns from (+-)id,creation_time,indicator")

get_parser_q = reqparse.RequestParser()
get_parser_q.add_argument('query', help="please enter (+-)integer or just integer")


#file_name is in 'xx.json' format
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
        if c.fetchall() != []:
            return df, max, None
    except:
        pass
    #select the max id
    try:
        c.execute('''select max(id) from collections''')
        max = c.fetchone()[0]
    except:
        #if there no exist max value, prove that the database is empty
        pass

    #get the time
    now = datetime.now()
    current_time = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    #store id and create_time
    for d in json_data:
        #in case the collections are all deleted
        if max == None:
            max = 0
        id = max + 1
        d['id'] = id
        url = "/collection/" + str(id)
        d['creation_time'] = current_time
        d['uri'] = url
        d['date'] = d['date'][0:4]
        d['date'] = int(d['date'])

        #or use df = pd.json_normalize()
        d['indicator.id'] = d['indicator']['id']
        d['indicator.value'] = d['indicator']['value']
        d['country.id'] = d['country']['id']
        d['country.value'] = d['country']['value']

        d.pop('indicator')
        d.pop('country')

    df = pd.DataFrame(data=json_data)
    #clear entires with nan
    df.dropna(inplace=True)

    #if all values are na
    return df, max+1, current_time

def store_df(df):
    cnn = sqlite3.connect("z5183932.db")
    sql.to_sql(df, name="collections", con=cnn, if_exists="append")
    cnn.close()

@api.response(400, 'Invalidation Error')
@api.route('/collections')
class collections(Resource):
    @api.response(201, 'created')
    @api.expect(post_parser)
    def post(self):
        args = post_parser.parse_args()
        indicator = args.get('indicator_id')

        #return 400 code when misssing id
        if indicator == None:
            return {"message": "Missing indicator id"}, 400

        #fetch the file
        data = fetch(indicator)

        #check if the indicator exist in source
        is_exist = False
        try:
            data[0]['message']
        except:
            is_exist = True

        if is_exist:
            #convert json to df and return the collection id
            df, id, current_time = json_to_df(data, indicator)
        else:
            return {"message": "id {} does not exist in source".format(indicator)}, 404
        
        #indicator already in local database
        if df.empty and current_time == None:
            return {"message": "id {} already exist in database".format(indicator)}, 404

        #all value are NA
        if df.empty:
            return {"message": "id {} has not valid value".format(indicator)}, 404

        #write to database
        store_df(df)
        url = "/collection/" + str(id)

        message = {
            'uri': url,
            'id': id,
            'creation_time': current_time,
            'indicator_id': indicator
        }
        return make_response(jsonify(message), 201)

    @api.response(200, 'OK')
    @api.expect(get_parser)
    def get(self):
        args = get_parser.parse_args()
        order_by = args.get('order_by')

        #check if arg exists
        if order_by == None:
            return {'meassage': "Missing argument"}, 400

        columns = order_by.split(',')
        num = len(columns)
        for index, c in enumerate(columns):
    
            #check the format
            if c[0] not in ["+", "-"]:
                return {'meassage': 'Invalid syntax, missing "+-"'}, 400

            #check if the column name is valid
            name = c[1:]
            if name not in ["id", "creation_time", "indicator"]:
                return {'meassage': "column: {} is not available for ordering".format(name)}, 400

            #fix name of indiccator
            if name == 'indicator':
                name = '[indicator.id]'
                c = c[0] + name
            #fix format for quering
            if c[0] == "+":
                c = c + " ASC"
            else:
                c = c + "DESC"

            columns[index] = c[1:]

        columns = ",".join(columns)

        cnn = sqlite3.connect('z5183932.db')
        c = cnn.cursor()
        c.execute('select uri, id, creation_time, [indicator.id] from collections group by id order by {}'.format(str(columns)))

        #get the collection info to a list
        values = []
        key = ('uri', 'id', 'creation_time', 'indicator')
        for value in c.fetchall():
            info = dict(zip(key, value))
            values.append(info)
        
        cnn.close()
        return make_response(jsonify(values),200)

@api.response(200, 'OK')
@api.response(404, 'Not found')
@api.route('/collections/<int:id>')
class collection(Resource):
    def delete(self, id):
        #check if the id exist(404 error)
        cnn = sqlite3.connect('z5183932.db')
        c = cnn.cursor()
        c.execute("select * from collections where id = \"{}\"".format(id))
        if c.fetchall() == []:
            cnn.close()
            return {"message": "id {} does not exist".format(id)}, 404
        
        #delete the data in database
        c.execute("delete from collections where id = {}".format(id))
        cnn.commit()
    
        message = "The collection " + str(id) + " was removed from the database"
        resp = {
            "message": message,
            "id": id
        }
        cnn.close()
        return make_response(jsonify(resp), 200)

    def get(self, id):
        cnn = sqlite3.connect('z5183932.db')
        c = cnn.cursor()
        c.execute("select [country.value], date, value from collections where id = \"{}\"".format(id))
        values = c.fetchall()
        if values == []:
            return {"message": "id {} does not exist".format(id)}, 404
        key = ("country", "date", "value")

        info = []
        for value in values:
            value = dict(zip(key, value))
            info.append(value)
        
        c.execute('select distinct id, [indicator.id], [indicator.value], creation_time from collections where id = {}'.format(id))
        key = ('id', 'indicator', 'indicator_value', 'creation_time')
        head = dict(zip(key, c.fetchone()))
        head['entries'] = info
        
        cnn.close()
        return make_response(jsonify(head), 200)

@api.response(200, 'OK')
@api.response(404, 'Not found')
@api.route('/collections/<int:id>/<int:year>/<string:country>')
class country(Resource):
    def get(self, id, year, country):
        cnn = sqlite3.connect('z5183932.db')
        c = cnn.cursor()

        c.execute('select distinct [indicator.id], [indicator.value] from collections where id = {}'.format(id))
        
        tmp = c.fetchone()
        if tmp == None:
            return {"message": "id {} does not exist".format(id)}, 404

        c.execute("select id, [indicator.id], [country.value], date, value from collections where id = {} and [country.value] = \"{}\" and date = {}".format(id, country, year))
        
        info = c.fetchone()
        if info == None:
            return {"message": " {} {} does not exist in collections/{}".format(year, country, id)}, 404

        key = ("id", "indicator", "country", "year", "value")
        message = dict(zip(key, info))

        return make_response(jsonify(message), 200)

@api.response(200, 'OK')
@api.response(404, 'Not found')
@api.response(400, 'Invalidation Error')
@api.route('/collections/<int:id>/<int:year>')
@api.expect(get_parser_q)
class year(Resource):
    def get(self, id, year):
        arg = get_parser_q.parse_args()
        query = arg.get('query')
        order_type = "not known"
        
        #check the format
        if query == None:
            return {"message": "Missing query number"},400

        #if + requested, use DESC; if - requested, use ASC
        if query[0] == "+":
            order_type = "DESC"
            num = query[1:]
        elif query[0] == "-":
            order_type = "ASC"
            num = query[1:]
        else:
            num = query
            order_type = "DESC"
        
        #get the number of line
        if not num.isdigit():
            return {"message": "invalid input format"},400
        num = int(num)
        #check range 1..100
        if num not in range(1,101):
            return {"message": "the number should between 1 and 100"},400

        #connect to database to get the tuple
        cnn = sqlite3.connect('z5183932.db')
        c = cnn.cursor()
        c.execute('select distinct [indicator.id], [indicator.value] from collections where id = {}'.format(id))
        
        tmp = c.fetchone()
        if tmp == None:
            return {"message": "id {} does not exist".format(id)}, 404

        key = ('indicator', 'indicator_value')
        message = dict(zip(key, tmp))


        c.execute('select [country.value], value from collections where id = {} and date = {} order by value {} limit {}'.format(id, year, order_type, num))
        entries = []
        key = ("country", "value")
        for item in c.fetchall():
            entries.append(dict(zip(key, item)))

        message['entries'] = entries

        cnn.close()
        return make_response(jsonify(message), 200)
if __name__=="__main__":
    app.run(debug=True)