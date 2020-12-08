import os
import json
import requests
import pandas as pd

from flask import Flask, request, Response

# Contants
# Documentation: https://core.telegram.org/bots/api
TOKEN = os.environ.get('TOKEN')

# # Info about the Bot
# https://api.telegram.org/bot{TOKEN}/getMe
		
# # Get Update
# https://api.telegram.org/bot{TOKEN}/getUpdates
#{"ok":true,"result":[{"update_id":571661109,
# "message":{"message_id":3,"from":{"id":1476417204,"is_bot":false,"first_name":"Fabio","last_name":"Caffarello","language_code":"pt-br"},"chat":{"id":1476417204,"first_name":"Fabio","last_name":"Caffarello","type":"private"},"date":1607390409,"text":"Oi amigo"}}]}

# # WebHook
# https://api.telegram.org/bot{TOKEN}/setWebhook?url=https://bot-rossmann-telegram.herokuapp.com/
		
# # Send Message
# https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id&text=Oi amigo!!
# https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id=1476417204&text=Oi amigo!!
#{"ok":true,"result":{"message_id":4,"from":{"id":1353530992,"is_bot":true,"first_name":"RossmannBot","username":"RossmannPredictBot"},"chat":{"id":1476417204,"first_name":"Fabio","last_name":"Caffarello","type":"private"},"date":1607390633,"text":"Oi amigo!!"}}



def sendMessage(chat_it, text):
	url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
	url += 'sendMessage?chat_id={}'.format(chat_it)
	
	r = requests.post(url, json={'text': text})
	print('Status Code {}'.format(r.status_code))
	
	return None



def loadDataset(storeId):
	# Loding test and store dataset
	testRaw = pd.read_csv('test.csv', low_memory=False)
	storeRaw = pd.read_csv('store.csv', low_memory=False)

	# Merge Test dataset + Store
	dfTest = pd.merge(testRaw, storeRaw, how='left', on='Store')

	# Choose Store for Prediction
	dfTest = dfTest[dfTest['Store'] == storeId]
	
	if not dfTest.empty:
		# Remove Closed Days
		dfTest = dfTest[dfTest['Open'] != 0]
		dfTest = dfTest[~dfTest['Open'].isnull()]
		dfTest = dfTest.drop('Id', axis=1)

		# Convert DataFrame to JSON
		data = json.dumps(dfTest.to_dict(orient='records'))
	else:
		data = 'error'
		
		
	return data



def predict(data):
	## API Call
	url = 'https://model-rossmann.herokuapp.com/rossmann/predict'
	header = {'Content-Type': 'application/json'}
	data = data

	r = requests.post(url, data=data, headers=header)
	#print('Status Code {}'.format(r.status_code))

	dfResponse = pd.DataFrame(r.json(), columns=r.json()[0].keys())
	
	return dfResponse



def parseMessage(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace( '/', '' )

    try:
        store_id = int(store_id)

    except ValueError:
        store_id = 'error'

    return chat_id, store_id

# API initialize
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		message = request.get_json()
		
		chat_id, store_id = parseMessage(message)
		
		if store_id != 'error':
			# Loading Data
			data = loadDataset(store_id)
			
			if data != 'error':
				# Prediction
				d1 = predict(data)

				# Calculate
				d2 = d1[['Store', 'Prediction']].groupby('Store').sum().reset_index()
				
				# Send Message
				msg = 'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(
							d2['Store'].values[0],
							d2['Prediction'].values[0])
				
				sendMessage(chat_id, msg)
				return Response('Ok', status=200)
				
			else:
				sendMessage(chat_id, 'Store Not Available')
				return Response('Ok', status=200)
			
		else:
			sendMessage(chat_id, 'Store Id is Wrong')
			return Response('Ok', status=200)
		
	else:
		return '<h1> Rossmann Telegram Bot </h1>'
		return Response('Ok', status=200)

if __name__ == '__main__':
	port = os.environ.get('PORT', 5000)
	app.run(host='0.0.0.0', port=port)


