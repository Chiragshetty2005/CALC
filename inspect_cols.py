import pandas as pd
import os
DF = pd.read_csv('heart_disease_uci_encoded_with_id.csv')
print('columns:', DF.columns.tolist())
row = DF[DF['id']==4].iloc[0]
print('\nrow index:')
for c in row.index:
    print(c, '->', row[c])
