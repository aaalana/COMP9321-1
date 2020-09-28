import pandas as pd
import q04

def new_df():
    df = q04.new_df()
    impact = []
    for index, row in df.iterrows():
        budget = row['budget']
        revenue = row['revenue']
        result = (revenue - budget)/budget
        if result < (-1.0):
            print(result)
            print(revenue)
            print(budget)
        impact.append(result)
    
    df['success_impact'] = impact
    return df

if __name__=='__main__':
    df = new_df()