import numpy as np
import pickle
import datetime 
import pandas as pd


def Ri(S):
    return([np.log(S[i + 1] / S[i]) for i in np.arange(0, np.size(S) - 1)])

def cleanStockData(df):
    df = df[['Data', 'Zamkniecie']]
    df['Data'] = df['Data'].apply(lambda x: datetime.date.fromisoformat(x))
    return df

def getParamsfromData(df1, df2=None, start_date=None, end_date=None):
    if start_date:
        df1 = df1.loc[df1['Data'] >= start_date, :]
        if df2 is not None:
            df2 = df2.loc[df2['Data'] >= start_date, :]
    if end_date:
        df1 = df1.loc[df1['Data'] < end_date, :]
        if df2 is not None:
            df2 = df2.loc[df2['Data'] < end_date, :]
    returns_1 = Ri(df1['Zamkniecie'].to_numpy())
    mu_1 = np.mean(returns_1) * const_num_day_year
    sigma_1 = np.std(returns_1, ddof=1) * const_num_day_year
    if df2 is not None:
        returns_2 = Ri(df2['Zamkniecie'].to_numpy())
        mu_2 = np.mean(returns_2) * const_num_day_year
        sigma_2 = np.std(returns_2, ddof=1) * const_num_day_year
        rho = np.corrcoef(returns_1, returns_2)[0][1]
        return {'Mu_1'   : mu_1,
                'Mu_2'   : mu_2,
                'Sigma_1': sigma_1,
                'Sigma_2': sigma_2,
                'rho'    : rho}
    return {'Mu_1': mu_1, 'Sigma_1': sigma_1}

const_num_day_year = 252
seed = 0
global_rng = np.random.default_rng(seed)

startDate1 = datetime.date.fromisoformat('2017-03-28')
startDate2 = datetime.date.fromisoformat('2017-10-01')

endDate1   = datetime.date.fromisoformat('2018-03-28')
endDate2   = datetime.date.fromisoformat('2018-10-01')

execDate  = datetime.date.fromisoformat('2018-12-21')

wig20 = cleanStockData(pd.read_csv("wig20_28.03.2017_31.12.2018.csv"))
kghm = cleanStockData(pd.read_csv("kghm_28.03.2017_31.12.2018.csv"))

params1 = getParamsfromData(wig20, kghm, startDate1, endDate1)
params2 = getParamsfromData(wig20, kghm, startDate2, endDate2)
