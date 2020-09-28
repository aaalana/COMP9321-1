import pandas as pd

def print_dataframe(dataframe, print_column=True, print_rows=True):
    # print column names
    if print_column:
        print(",".join([column for column in dataframe]))

    # print rows one by one
    if print_rows:
        for index, row in dataframe.iterrows():
            print(",".join([str(row[column]) for column in dataframe]))


if __name__=='__main__':
    csv_file = "Demographic Statistics.csv"
    #load csv
    dataframe = pd.read_csv(csv_file)
    print("loading the cvs file")
    print_dataframe(dataframe)

    #write
    dataframe.to_csv('Demographic Statistics_NEW.csv', sep=',', encoding='utf-8')

