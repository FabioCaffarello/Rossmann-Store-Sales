import json
import requests
import pandas as pd


def loadDataset(storeId):
	# Loding test and store dataset
	testRaw = pd.read_csv('../../01-Data/test.csv', low_memory=False)
	storeRaw = pd.read_csv('../../01-Data/store.csv', low_memory=False)

	# Merge Test dataset + Store
	dfTest = pd.merge(testRaw, storeRaw, how='left', on='Store')

	# Choose Store for Prediction
	dfTest = dfTest[dfTest['Store'] == storeId]

	# Remove Closed Days
	dfTest = dfTest[dfTest['Open'] != 0]
	dfTest = dfTest[~dfTest['Open'].isnull()]
	dfTest = dfTest.drop('Id', axis=1)

	# Convert DataFrame to JSON
	data = json.dumps(dfTest.to_dict(orient='records'))
	
	return data



def predict(data):
	## API Call
	url = 'https://model-rossmann.herokuapp.com/rossmann/predict'
	header = {'Content-Type': 'application/json'}
	data = data

	r = requests.post(url, data=data, headers=header)
	print('Status Code {}'.format(r.status_code))

	dfResponse = pd.DataFrame(r.json(), columns=r.json()[0].keys())
	
	return dfResponse


# dfResponse2 = dfResponse[['Store', 'Prediction']].groupby('Store').sum().reset_index()

# for i in range(len(dfResponse2)):
#     print('Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(
#             dfResponse2.loc[i, 'Store'],
#              dfResponse2.loc[i, 'Prediction']))