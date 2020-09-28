import pandas as pd
import requests
def fetch(url):
    params = dict(
        origin='Chicago,IL',
        destination='Los+Angeles,CA',
        waypoints='Joplin,MO|Oklahoma+City,OK',
        sensor='false'
    )
    resp = requests.get(url=url, params=params)
    data = resp.json()
    return data

def json_to_dataframe(json_obj):
    json_data = json_obj['data']
    print(json_data)

    columns = []
    for c in json_obj['meta']['view']['columns']:
        columns.append(c['name'])

    return pd.DataFrame(data=json_data, columns=columns)

def print_dataframe(dataframe):
    
    print(",".join(coloumn for coloumn in dataframe))

    for index, row in dataframe.iterrows():
        print(",".join([str(row[column]) for column in dataframe]))

    
if __name__=='__main__':
    csv_file = 'Demographic Statistics.csv'
    dataframe = pd.read_csv(csv_file)

    #fetch the url
    url = "https://data.cityofnewyork.us/api/views/kku6-nxdu/rows.json"
    json_obj = fetch(url)
    dataframe = json_to_dataframe(json_obj)
    #print_dataframe(dataframe)