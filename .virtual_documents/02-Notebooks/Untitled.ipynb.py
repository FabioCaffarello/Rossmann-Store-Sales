import pandas as pd


salesRaw = pd.read_csv('../01-Data/train.csv', low_memory=False)
storeRaw = pd.read_csv('../01-Data/store.csv', low_memory=False)


dfRaw = salesRaw.merge(storeRaw, how='left', on='Store')


dfRaw1 = dfRaw.copy()


dfRaw1.columns


print(f'Number of Rows: {dfRaw1.shape[0]}')
print(f'Number of Columns: {dfRaw1.shape[1]}')


dfRaw1.dtypes


dfRaw1.isnull().sum()


dfRaw1.isnull().mean()














import inflection
lista = ['oi_amigo', 'teste_de camel']

camelCase = lambda x: inflection.camelize(x)


list(map(camelCase, lista))



