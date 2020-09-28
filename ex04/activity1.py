
import matplotlib.pyplot as plt
import pandas as pd


def clean(dataframe):
    dataframe['Place of Publication'] = dataframe['Place of Publication'].apply(
        lambda x: 'London' if 'London' in x else x.replace('-', ' '))

    new_date = dataframe['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
    new_date = pd.to_numeric(new_date)
    new_date = new_date.fillna(0)
    dataframe['Date of Publication'] = new_date

    return dataframe

if __name__ == '__main__':
    csv_file = 'Books.csv'
    df = pd.read_csv(csv_file)

    # Cleaning is Optional; but it will increase the accuracy of the results
    df = clean(df)
    print(df['Place of Publication'].to_string())

    # value_counts: returns a Series containing counts of each category.
    unival = df['Place of Publication'].value_counts()
    unival.plot.pie(subplots=True)

    plt.show()