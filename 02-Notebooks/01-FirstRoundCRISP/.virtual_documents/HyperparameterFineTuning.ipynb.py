import pandas as pd
import numpy as np
import random
import datetime
import pickle

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import HTML

from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso

import xgboost as xgb
from lightgbm import LGBMRegressor


def crossValidation(XTraining, kfold, modelName, model='default', verbose=False):
    maeList = []
    mapeList = []
    rmseList = []

    for k in reversed(range(1, kfold+1)):
        if verbose:
            print(f'\nKFold Number: {k}')
        # Start and End Date for Validation
        startDateValid = XTraining['Date'].max() - datetime.timedelta(days=k*6*7)
        endDateValid = XTraining['Date'].max() - datetime.timedelta(days=(k-1)*6*7)

        # Filtering Dataset
        training = XTraining[XTraining['Date'] < startDateValid]
        validation = XTraining[(XTraining['Date'] >= startDateValid) & (XTraining['Date'] <= endDateValid)]

        # Training and Validation Dataset
        # Training
        XKFoldTraining = training.drop(['Date', 'Sales'], axis=1)
        yKFoldTraining = training['Sales']

        # Validation
        XKFoldValidation = validation.drop(['Date', 'Sales'], axis=1)
        yKFoldValidation = validation['Sales']

        # Model
        ## Model Map
        modelMap = {
            'Linear Regression': LinearRegression(),
            'Lasso': Lasso(alpha=0.01),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
            'XGBoost Regressor': xgb.XGBRegressor( objective='reg:squarederror', n_estimators=500, eta=0.01, max_depth=10, 
                                                      subsample=0.7, colsample_bytree=0.9),
            'Lightgbm Regressor': LGBMRegressor(num_leaves=10, min_data_in_leaf=50, n_jobs=-1, random_state=42, n_estimators=500)   
        }
        
        ## Mapped Model
        if model == 'default':
            model = modelMap[modelName]
        else: model = model
        
        model.fit(XKFoldTraining, yKFoldTraining)

        # Prediction
        yhat = model.predict(XKFoldValidation)

        #Performance
        modelResult = mlError('Linear Regression', np.expm1(yKFoldValidation), np.expm1(yhat))
        
        #Store Performance of each KFold iteration
        maeList.append(modelResult['MAE'].tolist())
        mapeList.append(modelResult['MAPE'].tolist())
        rmseList.append(modelResult['RMSE'].tolist())


    dictResult = {
                    'Model Name': [modelName],
                    'MAE CV': [np.round(np.mean(maeList),2).astype(str) + ' +/- ' + np.round(np.std(maeList),2).astype(str)],
                    'MAPE CV': [np.round(np.mean(mapeList),2).astype(str) + ' +/- ' + np.round(np.std(mapeList),2).astype(str)],
                    'RMSE CV': [np.round(np.mean(rmseList),2).astype(str) + ' +/- ' + np.round(np.std(rmseList),2).astype(str)]
                }

    return pd.DataFrame(dictResult)


def mean_percentage_error( y, yhat ):
    return np.mean( ( y - yhat ) / y )

def mean_absolute_percentage_error(y, yhat):
    return np.mean(np.abs((y - yhat) / y))


def mlError(modelName, y, yhat):
    mae = mean_absolute_error(y, yhat)
    mape = mean_absolute_percentage_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    
    return pd.DataFrame({
                            'ModelName': modelName,
                            'MAE': mae,
                            'MAPE': mape,
                            'RMSE': rmse,
                        }, index=[0])



def jupyter_settings():
    get_ipython().run_line_magic("matplotlib", " inline")
    get_ipython().run_line_magic("pylab", " inline")
    
    plt.style.use( 'bmh' )
    plt.rcParams['figure.figsize'] = [25, 16]
    plt.rcParams['font.size'] = 24
    
    display( HTML( '<style>.container { width:100% get_ipython().getoutput("important; }</style>') )")
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option( 'display.expand_frame_repr', False )
    
    sns.set()


jupyter_settings()


dfRaw = pd.read_csv('../../01-Data/Results/01-FirstRoundCRISP/dfDataPreparation.csv', low_memory=False, parse_dates=['Date'])


dfRaw1 = dfRaw.copy()


toKeepBoruta = [
                'Store',
                'Promo',
                'StoreType',
                'Assortment',
                'CompetitionDistance',
                'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear',
                'Promo2',
                'Promo2SinceWeek',
                'Promo2SinceYear',
                'CompetionTimeMonth',
                'PromoTimeWeek',
                'MonthSin',
                'MonthCos',
                'DaySin',
                'DayCos',
                'WeekOfYearSin',
                'WeekOfYearCos',
                'DayOfWeekSin',
                'DayOfWeekCos',
                'Date',
                'Sales']


toKeep = toKeepBoruta[:-2]

#Training Dataset
XTrain = dfRaw1[dfRaw1['Date'] < '2015-06-19']
yTrain = XTrain['Sales']
XTr = XTrain[toKeep]


#Validation Dataset
XTest = dfRaw1[dfRaw1['Date'] >= '2015-06-19']
yTest = XTest['Sales']
XTst = XTest[toKeep]


#Training Dataset
XTraining = XTrain[toKeepBoruta]


param = {
   'n_estimators': [1500, 1700, 2500, 3000, 3500],
   'eta': [0.01, 0.03],
   'max_depth': [3, 5, 9],
   'subsample': [0.1, 0.5, 0.7],
   'colsample_bytree': [0.3, 0.7, 0.9],
   'min_child_weight': [3, 8, 15]
       }

MAX_EVAL = 5


finalResult = pd.DataFrame()

for i in range( MAX_EVAL ):
    # choose values for parameters randomly
    hp = { k: random.sample( v, 1 )[0] for k, v in param.items() }
    print(hp)

    # model
    modelXGBR = xgb.XGBRegressor( objective='reg:squarederror',
                                 n_estimators=hp['n_estimators'], 
                                 eta=hp['eta'], 
                                 max_depth=hp['max_depth'], 
                                 subsample=hp['subsample'],
                                 colsample_bytree=hp['colsample_bytree'],
                                 min_child_weight=hp['min_child_weight'])

    # performance
    result = crossValidation( XTraining, 5, 'XGBoost Regressor', modelXGBR, verbose=True )
    finalResult = pd.concat( [finalResult, result] )

finalResult


paramTuned = {
                'n_estimators': 2500,
                'eta': 0.01,
                'max_depth': 9,
                'subsample': 0.1,
                'colsample_bytree': 0.7,
                'min_child_weight': 8
            }


# model
modelXGBRTuned = xgb.XGBRegressor(objective='reg:squarederror',
                                    n_estimators=paramTuned['n_estimators'], 
                                    eta=paramTuned['eta'], 
                                    max_depth=paramTuned['max_depth'], 
                                    subsample=paramTuned['subsample'],
                                    colsample_bytree=paramTuned['colsample_bytree'],
                                    min_child_weight=paramTuned['min_child_weight'])

modelXGBRTuned.fit(XTr, yTrain)

# prediction
yhatXGBTuned = modelXGBRTuned.predict(XTst)

# performance
modelXGBRTunedResult = mlError( 'XGBoost Regressor', np.expm1(yTest), np.expm1(yhatXGBTuned))
modelXGBRTunedResult


mpe = mean_percentage_error(np.expm1(yTest), np.expm1( yhatXGBTuned))
mpe


dfFinalModel = XTest[toKeepBoruta]

# rescale
dfFinalModel['Sales'] = np.expm1(dfFinalModel['Sales'])
dfFinalModel['Predictions'] = np.expm1(yhatXGBTuned)


dfFinalModel.to_csv('../../01-Data/Results/01-FirstRoundCRISP/dfFinalModel.csv', index=False)


pickle.dump(modelXGBRTuned, open('D:/01-DataScience/04-Projetos/00-Git/Rossmann-Store-Sales/02-Notebooks/01-FirstRoundCRISP/model/modelRossmann.pkl', 'wb' ))
