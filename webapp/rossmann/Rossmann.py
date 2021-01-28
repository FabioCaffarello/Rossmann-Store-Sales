import pickle
import pandas as pd
import numpy as np
import math
import inflection
import datetime




class Rossmann(object):
	def __init__(self):
		self.home_path = ''
		self.competition_distance_scaler = pickle.load(open(self.home_path + 'parameter/1-competition_distance_scaler.pkl', 'rb'))
		self.competion_time_month_scaler = pickle.load(open(self.home_path + 'parameter/1-competion_time_month_scaler.pkl', 'rb'))
		self.promo_time_week_scaler      = pickle.load(open(self.home_path + 'parameter/1-promo_time_week_scaler.pkl', 'rb'))
		self.year_scaler                 = pickle.load(open(self.home_path + 'parameter/1-year_scaler.pkl', 'rb'))
		self.store_type_scaler           = pickle.load(open(self.home_path + 'parameter/1-store_type_scaler.pkl', 'rb'))
		
		
	def rename_columns(self, df1):
		# snakecase
		snakecase = lambda col: inflection.underscore(col)
		new_columns = list(map(snakecase, df1.columns))

		# rename
		df1.columns = new_columns
		return df1
		
	

	def data_cleaning(self, df1):

		df1 = Rossmann.rename_columns(self, df1)

		## Data Types
		df1['date'] = pd.to_datetime(df1['date'])

		## Fillout NA
		# competition_distance
			#distance in meters to the nearest competitor store
		df1['competition_distance'] = df1['competition_distance'].apply(lambda row: 200000.0 if math.isnan(row) else row)


		# competition_open_since_month
			#gives the approximate month of the time the nearest competitor was opened
		df1['competition_open_since_month'] = df1.apply(lambda row: row['date'].month if math.isnan(row['competition_open_since_month']) else row['competition_open_since_month'], axis=1)


		# CompetitionOpenSinceYear
			# gives the approximate year of the time the nearest competitor was opened
		df1['competition_open_since_year'] = df1.apply(lambda row: row['date'].year if math.isnan(row['competition_open_since_year']) else row['competition_open_since_year'], axis=1)


		# promo2_since_week Date
			#describes the calendar week when the store started participating in Promo2
		df1['promo2_since_week'] = df1.apply(lambda row: row['date'].week if math.isnan(row['promo2_since_week']) else row['promo2_since_week'], axis=1)


		# promo2_since_year
			#describes the year when the store started participating in Promo2
		df1['promo2_since_year'] = df1.apply(lambda row: row['date'].year if math.isnan(row['promo2_since_year']) else row['promo2_since_year'], axis=1)


		# promo_interval
			#describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew.\
			#E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
		month_map = {
						1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
					}

		df1['promo_interval'].fillna(0, inplace=True)
		df1['month_map'] = df1['date'].dt.month.map(month_map)

		df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda row: 0 if row['promo_interval'] == 0 else 1 if row['month_map'] in row['promo_interval'].split(',') else 0, axis=1)

		# competiton
		df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
		df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

		# promo2
		df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
		df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

		return df1
	
	
	
	
	def feature_engineering(self, df2):

		#year
		df2['year'] = df2['date'].dt.year

		#month
		df2['month'] = df2['date'].dt.month

		#day
		df2['day'] = df2['date'].dt.day

		#week of year
		df2['week_of_year'] = df2['date'].dt.weekofyear

		#year week
		df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

		#Competion Sinse
		df2['competion_sinse'] = df2.apply(lambda row: datetime.datetime(year=row['competition_open_since_year'], month=row['competition_open_since_month'], day=1), axis=1)
		df2['competion_time_month'] = ((df2['date'] - df2['competion_sinse'])/30).apply(lambda row: row.days).astype(int)

		#Promo Since
		df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
		df2['promo_since'] = df2['promo_since'].apply(lambda row: datetime.datetime.strptime(row + '-1',  '%Y-%W-%w') - datetime.timedelta(days=7))
		df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda row: row.days).astype(int)

		#Assortment (level: a = basic, b = extra, c = extended)
		level = {
			'a' : 'basic', 'b' : 'extra', 'c' : 'extended'
		}
		df2['assortment'] = df2['assortment'].map(level)

		# State Holiday (a = public holiday, b = Easter holiday, c = Christmas, 0 = None)
		holiday = {
			'a' : 'public holiday', 'b' : 'Easter holiday', 'c' : 'Christmas'
		}
		df2['state_holiday'] = df2['state_holiday'].map(holiday)
		df2['state_holiday'].fillna('regular day', inplace=True)

		## Row Fitering
		df2 = df2[df2['open'] != 0]

		## Columns Filtering
		toDrop = ['open', 'promo_interval', 'month_map']
		df2.drop(toDrop, axis=1, inplace=True)

		return df2
	
	

	def data_preparation(self, df3):

		#Competion Distance >> Presence of well defined outiliers
		df3['competition_distance'] = self.competition_distance_scaler.fit_transform(df3[['competition_distance']].values)

		#Competion Time Month >> Presence of well defined outiliers
		df3['competion_time_month'] = self.competion_time_month_scaler.fit_transform(df3[['competion_time_month']].values)

		#Promo Time Week
		df3['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df3[['promo_time_week']].values)

		#Year
		df3['year'] = self.year_scaler.fit_transform(df3[['year']].values)

		### Encoding
		#State Holiday -> One Hot Encoding
		df3 = pd.get_dummies(df3, prefix=['state_holiday'], columns=['state_holiday'])

		#Store Type -> Label Encoding
		df3['store_type'] = self.store_type_scaler.fit_transform(df3['store_type'])

		#Assortment -> Ordinal Encoding
		dict_assortment = {
							'basic': 1,
							'extra': 2,
							'extended': 3
							}
		df3['assortment'] = df3['assortment'].map(dict_assortment)

		### Nature Transformation
		#Month
		df3['month_sin'] = df3['month'].apply(lambda row: np.sin(row * (2 * np.pi/12)))
		df3['month_cos'] = df3['month'].apply(lambda row: np.cos(row * (2 * np.pi/12)))
		#Day
		df3['day_sin'] = df3['day'].apply(lambda row: np.sin(row * (2 * np.pi/30)))
		df3['day_cos'] = df3['day'].apply(lambda row: np.cos(row * (2 * np.pi/30)))
		#Week of Year
		df3['week_of_year_sin'] = df3['week_of_year'].apply(lambda row: np.sin(row * (2 * np.pi/52)))
		df3['week_of_year_cos'] = df3['week_of_year'].apply(lambda row: np.cos(row * (2 * np.pi/52)))
		#Day of Week
		df3['day_of_week_sin'] = df3['day_of_week'].apply(lambda row: np.sin(row * (2 * np.pi/7)))
		df3['day_of_week_cos'] = df3['day_of_week'].apply(lambda row: np.cos(row * (2 * np.pi/7)))

		cols_selected = ['store','promo','store_type','assortment','competition_distance','competition_open_since_month',
								'competition_open_since_year','promo2','promo2_since_week','promo2_since_year','competion_time_month',
								'promo_time_week','month_sin','month_cos','day_sin','day_cos','week_of_year_sin','week_of_year_cos',
								'day_of_week_sin','day_of_week_cos']

		return df3[cols_selected]


	
	def get_prediction(self, model, original_data, test_data):
		# Prediction
		pred = model.predict(test_data)

		# Join pred into original Data
		original_data['prediction'] = np.expm1(pred)

		return original_data.to_json(orient='records', date_format='iso')