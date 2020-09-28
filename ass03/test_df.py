import pandas as pd

if __name__=='__main__':
    train = pd.read_csv('validation.csv')
    df = pd.DataFrame(columns=['movie_id', 'predicted_revenue'])
    df['movie_id'] = [1]
    df['predicted_revenue'] = pd.Series(['bal', 'hi'])
    df.set_index('movie_id', inplace=True)
    df.to_csv('z5183932.PART1.summary.csv')
    print(df)
    
