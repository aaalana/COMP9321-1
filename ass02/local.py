import json
import urllib.request, urllib.error, urllib.parse



if __name__=="__main__":
    url = "http://api.worldbank.org/v2/countries/all/indicators/1.1.PSev.Poor4uds?date=2012:2017&format=json&per_page=1000"
    response = urllib.request.urlopen(url)
    webContent = response.read()
    print(webContent)
    f = open('1.1.PSev.Poor4uds.html', 'wb')
    f.write(webContent)
    f.close
