import pandas as pd
import math


salesRaw = pd.read_csv('../01-Data/train.csv', low_memory=False)
storeRaw = pd.read_csv('../01-Data/store.csv', low_memory=False)


dfRaw = salesRaw.merge(storeRaw, how='left', on='Store')


dfRaw1 = dfRaw.copy()


dfRaw1.columns


print(f'Number of Rows: {dfRaw1.shape[0]}')
print(f'Number of Columns: {dfRaw1.shape[1]}')


dfRaw1.dtypes


dfRaw1['Date'] = pd.to_datetime(dfRaw1['Date'])


dfRaw1.isnull().sum()


dfRaw1.isnull().mean()


maxValueCompetitionDistance = dfRaw1['CompetitionDistance'].max()


dfRaw1.sample(5)


# CompetitionDistance
    #distance in meters to the nearest competitor store
maxValueCompetitionDistance = dfRaw1['CompetitionDistance'].max()
dfRaw1['CompetitionDistance'] = dfRaw1['CompetitionDistance'].apply(lambda row: maxValueCompetitionDistance*100 if math.isnan(row) else row)


# CompetitionOpenSinceMonth
    #gives the approximate month of the time the nearest competitor was opened
dfRaw1['CompetitionOpenSinceMonth'] = dfRaw1.apply(lambda row: row['Date'].month if math.isnan(row['CompetitionOpenSinceMonth']) else row['CompetitionOpenSinceMonth'], axis=1)


# CompetitionOpenSinceYear
    # gives the approximate year of the time the nearest competitor was opened
dfRaw1['CompetitionOpenSinceYear'] = dfRaw1.apply(lambda row: row['Date'].year if math.isnan(row['CompetitionOpenSinceYear']) else row['CompetitionOpenSinceYear'], axis=1)


# Promo2SinceWeek
    #describes the calendar week when the store started participating in Promo2
dfRaw1['Promo2SinceWeek'] = dfRaw1.apply(lambda row: row['Date'].week if math.isnan(row['Promo2SinceWeek']) else row['Promo2SinceWeek'], axis=1)


# Promo2SinceYear
    #describes the year when the store started participating in Promo2
dfRaw1['Promo2SinceYear'] = dfRaw1.apply(lambda row: row['Date'].week if math.isnan(row['Promo2SinceYear']) else row['Promo2SinceYear'], axis=1)


# PromoInterval
    #describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew.\
    #E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
monthMap = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }

dfRaw1['PromoInterval'].fillna(0, inplace=True)
dfRaw1['MonthMap'] = dfRaw1['Date'].dt.month.map(monthMap)

dfRaw1['IsPromo'] = dfRaw1[['PromoInterval', 'MonthMap']].apply(lambda row: 0 if row['PromoInterval'] == 0 else 1 if row['MonthMap'] in row['PromoInterval'].split(',') else 0, axis=1)


dfRaw1.dtypes


dfRaw1['CompetitionOpenSinceMonth'] = dfRaw1['CompetitionOpenSinceMonth'].astype(int)
dfRaw1['CompetitionOpenSinceYear'] = dfRaw1['CompetitionOpenSinceYear'].astype(int)
dfRaw1['Promo2SinceWeek'] = dfRaw1['Promo2SinceWeek'].astype(int)
dfRaw1['Promo2SinceYear'] = dfRaw1['Promo2SinceYear'].astype(int)



