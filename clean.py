import pandas as pd

data = pd.read_csv('alexa_data.csv', sep='\t')

cleaned_data = data.drop(['feedback', 'date', 'variation'], axis=1)
print(cleaned_data)

cleaned_data.to_csv('cleaned_data.csv', index_label=None)





