import pickle
import pandas as pd
import numpy as np
import math
import datetime



class Rossmann(object):
	def __init__(self):
		self.home_path = 'D:/01-DataScience/04-Projetos/00-Git/Rossmann-Store-Sales/02-Notebooks/01-FirstRoundCRISP/'
		self.competitionDistanceScaler = pickle.load(open(self.home_path + 'parameter/CompetitionDistanceScaler.pkl', 'rb'))
		self.competionTimeMonthScaler =  pickle.load(open(self.home_path + 'parameter/CompetionTimeMonthScaler.pkl', 'rb'))
		self.promoTimeWeekScaler =       pickle.load(open(self.home_path + 'parameter/PromoTimeWeekScaler.pkl', 'rb'))
		self.yearScaler =                pickle.load(open(self.home_path + 'parameter/YearScaler.pkl', 'rb'))
		self.storeTypeScaler =           pickle.load(open(self.home_path + 'parameter/StoreTypeScaler.pkl', 'rb'))
		
		
		
		
	def dataCleaning(self, df1):

		## Data Types
		df1['Date'] = pd.to_datetime(df1['Date'])

		## Fillout NA
		# CompetitionDistance
			#distance in meters to the nearest competitor store
		df1['CompetitionDistance'] = df1['CompetitionDistance'].apply(lambda row: 200000.0 if math.isnan(row) else row)


		# CompetitionOpenSinceMonth
			#gives the approximate month of the time the nearest competitor was opened
		df1['CompetitionOpenSinceMonth'] = df1.apply(lambda row: row['Date'].month if math.isnan(row['CompetitionOpenSinceMonth']) else row['CompetitionOpenSinceMonth'], axis=1)


		# CompetitionOpenSinceYear
			# gives the approximate year of the time the nearest competitor was opened
		df1['CompetitionOpenSinceYear'] = df1.apply(lambda row: row['Date'].year if math.isnan(row['CompetitionOpenSinceYear']) else row['CompetitionOpenSinceYear'], axis=1)


		# Promo2SinceWeek
			#describes the calendar week when the store started participating in Promo2
		df1['Promo2SinceWeek'] = df1.apply(lambda row: row['Date'].week if math.isnan(row['Promo2SinceWeek']) else row['Promo2SinceWeek'], axis=1)


		# Promo2SinceYear
			#describes the year when the store started participating in Promo2
		df1['Promo2SinceYear'] = df1.apply(lambda row: row['Date'].year if math.isnan(row['Promo2SinceYear']) else row['Promo2SinceYear'], axis=1)


		# PromoInterval
			#describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew.\
			#E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
		monthMap = {
						1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
					}

		df1['PromoInterval'].fillna(0, inplace=True)
		df1['MonthMap'] = df1['Date'].dt.month.map(monthMap)

		df1['IsPromo'] = df1[['PromoInterval', 'MonthMap']].apply(lambda row: 0 if row['PromoInterval'] == 0 else 1 if row['MonthMap'] in row['PromoInterval'].split(',') else 0, axis=1)

		# competiton
		df1['CompetitionOpenSinceMonth'] = df1['CompetitionOpenSinceMonth'].astype(int)
		df1['CompetitionOpenSinceYear'] = df1['CompetitionOpenSinceYear'].astype(int)

		# promo2
		df1['Promo2SinceWeek'] = df1['Promo2SinceWeek'].astype(int)
		df1['Promo2SinceYear'] = df1['Promo2SinceYear'].astype(int)

		return df1
	
	
	
	
	def featureEngineering(self, df2):

		#year
		df2['Year'] = df2['Date'].dt.year

		#month
		df2['Month'] = df2['Date'].dt.month

		#day
		df2['Day'] = df2['Date'].dt.day

		#week of year
		df2['WeekOfYear'] = df2['Date'].dt.weekofyear

		#year week
		df2['YearWeek'] = df2['Date'].dt.strftime('%Y-%W')

		#Competion Sinse
		df2['CompetionSinse'] = df2.apply(lambda row: datetime.datetime(year=row['CompetitionOpenSinceYear'], month=row['CompetitionOpenSinceMonth'], day=1), axis=1)
		df2['CompetionTimeMonth'] = ((df2['Date'] - df2['CompetionSinse'])/30).apply(lambda row: row.days).astype(int)

		#Promo Since
		df2['PromoSince'] = df2['Promo2SinceYear'].astype(str) + '-' + df2['Promo2SinceWeek'].astype(str)
		df2['PromoSince'] = df2['PromoSince'].apply(lambda row: datetime.datetime.strptime(row + '-1',  '%Y-%W-%w') - datetime.timedelta(days=7))
		df2['PromoTimeWeek'] = ((df2['Date'] - df2['PromoSince'])/7).apply(lambda row: row.days).astype(int)

		#Assortment (level: a = basic, b = extra, c = extended)
		level = {
			'a' : 'basic', 'b' : 'extra', 'c' : 'extended'
		}
		df2['Assortment'] = df2['Assortment'].map(level)

		# State Holiday (a = public holiday, b = Easter holiday, c = Christmas, 0 = None)
		holiday = {
			'a' : 'public holiday', 'b' : 'Easter holiday', 'c' : 'Christmas'
		}
		df2['StateHoliday'] = df2['StateHoliday'].map(holiday)
		df2['StateHoliday'].fillna('Regular Day', inplace=True)

		## Row Fitering
		df2 = df2[df2['Open'] != 0]

		## Columns Filtering
		toDrop = ['Open', 'PromoInterval', 'MonthMap']
		df2.drop(toDrop, axis=1, inplace=True)

		return df2
	
	
	

	def dataPreparation(self, df3):

		#Competion Distance >> Presence of well defined outiliers
		df3['CompetitionDistance'] = self.competitionDistanceScaler.fit_transform(df3[['CompetitionDistance']].values)

		#Competion Time Month >> Presence of well defined outiliers
		df3['CompetionTimeMonth'] = self.competionTimeMonthScaler.fit_transform(df3[['CompetionTimeMonth']].values)

		#Promo Time Week
		df3['PromoTimeWeek'] = self.promoTimeWeekScaler.fit_transform(df3[['PromoTimeWeek']].values)

		#Year
		df3['Year'] = self.yearScaler.fit_transform(df3[['Year']].values)

		### Encoding
		#State Holiday -> One Hot Encoding
		df3 = pd.get_dummies(df3, prefix=['StateHoliday'], columns=['StateHoliday'])

		#Store Type -> Label Encoding
		df3['StoreType'] = self.storeTypeScaler.fit_transform(df3['StoreType'])

		#Assortment -> Ordinal Encoding
		dictAssortment = {
							'basic': 1,
							'extra': 2,
							'extended': 3
							}
		df3['Assortment'] = df3['Assortment'].map(dictAssortment)

		### Nature Transformation
		#Month
		df3['MonthSin'] = df3['Month'].apply(lambda row: np.sin(row * (2 * np.pi/12)))
		df3['MonthCos'] = df3['Month'].apply(lambda row: np.cos(row * (2 * np.pi/12)))
		#Day
		df3['DaySin'] = df3['Day'].apply(lambda row: np.sin(row * (2 * np.pi/30)))
		df3['DayCos'] = df3['Day'].apply(lambda row: np.cos(row * (2 * np.pi/30)))
		#Week of Year
		df3['WeekOfYearSin'] = df3['WeekOfYear'].apply(lambda row: np.sin(row * (2 * np.pi/52)))
		df3['WeekOfYearCos'] = df3['WeekOfYear'].apply(lambda row: np.cos(row * (2 * np.pi/52)))
		#Day of Week
		df3['DayOfWeekSin'] = df3['DayOfWeek'].apply(lambda row: np.sin(row * (2 * np.pi/7)))
		df3['DayOfWeekCos'] = df3['DayOfWeek'].apply(lambda row: np.cos(row * (2 * np.pi/7)))

		colsSelected = ['Store','Promo','StoreType','Assortment','CompetitionDistance','CompetitionOpenSinceMonth',
								'CompetitionOpenSinceYear','Promo2','Promo2SinceWeek','Promo2SinceYear','CompetionTimeMonth',
								'PromoTimeWeek','MonthSin','MonthCos','DaySin','DayCos','WeekOfYearSin','WeekOfYearCos','DayOfWeekSin',
								'DayOfWeekCos']

		return df3[colsSelected]


	
	def getPrediction(self, model, originalData, testData):
		# Prediction
		pred = model.predict(testData)

		# Join pred into original Data
		originalData['Prediction'] = np.expm1(pred)

		return originalData.to_json(orient='records', date_format='iso')