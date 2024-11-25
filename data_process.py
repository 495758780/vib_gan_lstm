import pandas as pd

df = pd.read_excel('data/meg.xlsx')
df_list = []

for i in range(1, 21):
    temp_df = df.iloc[12000+1000*i:13000+1000*i, :]
    temp_df.to_csv(f'data/gan/input/{i}.csv')