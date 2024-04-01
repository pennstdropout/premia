# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import dask.dataframe as dd
from dask.distributed import Client
import wrds
import os
from statsmodels.sandbox.regression import gmm
from statsmodels.api import OLS


# %%
# Import data

def request_wrds(path: str,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    with wrds.Connection() as db:
        df_s12 = db.raw_sql(f'''
            SELECT *
            FROM tfn.s12
            WHERE fdate >= '{start_date}' AND fdate <= '{end_date}'
        ''', date_cols=['fdate'])
        df_s12.to_csv(f'{path}s12.csv')

        df_s12type5 = db.raw_sql(f'''
                    SELECT *
                    FROM tfn.s12type5
                    WHERE fdate >= '{start_date}' AND fdate <= '{end_date}'
                ''', date_cols=['fdate'])
        df_s12type5.to_csv(f'{path}df_s12type5.csv')

        df_s34 = db.raw_sql(f'''
            SELECT *
            FROM tfn.s34
            WHERE fdate >= '{start_date}' AND fdate <= '{end_date}'
        ''', date_cols=['fdate'])
        df_s34.to_csv(f'{path}s34.csv')

        df_beta = db.raw_sql(f'''
            SELECT *
            FROM beta.ff3
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        ''', date_cols=['date'])
        df_beta.to_csv(f'{path}security.csv')

        df_security = db.raw_sql(f'''
            SELECT *
            FROM crspm.crspm_stock
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        ''', date_cols=['date'])
        df_security.to_csv(f'{path}security.csv')

        return df_s12, df_s34, df_security


def dask_read(path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    try:
        df_s12 = dd.read_csv(f'{path}s12.csv', encoding='ISO-8859-1')
        log_import_s12(df_s12)

        df_s12type5 = dd.read_csv(f'{path}s12type5.csv', encoding='ISO-8859-1')
        log_import_s12type5(df_s12type5)

        df_s34 = dd.read_csv(f'{path}s34.csv', encoding='ISO-8859-1')
        log_import_s34(df_s34)

        df_beta = dd.read_csv(f'{path}beta.csv', encoding='ISO-8859-1')
        log_import_beta(df_beta)

        df_security = dd.read_csv(f'{path}security.csv', encoding='ISO-8859-1')
        log_import_security(df_security)

        return df_s12, df_s12type5, df_s34, df_beta, df_security

    except FileNotFoundError:
        return request_wrds(path, start_date, end_date)


def pandas_read(path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    try:
        df_s12 = pd.read_csv(f'{path}s12.csv', encoding='ISO-8859-1', low_memory=False)
        log_import_s12(df_s12)

        df_s12type5 = pd.read_csv(f'{path}s12type5.csv', encoding='ISO-8859-1', low_memory=False)
        log_import_s12type5(df_s12type5)

        df_s34 = pd.read_csv(f'{path}s34.csv', encoding='ISO-8859-1', low_memory=False)
        log_import_s34(df_s34)

        df_beta = pd.read_csv(f'{path}beta.csv', encoding='ISO-8859-1', low_memory=False)
        log_import_beta(df_beta)

        df_security = pd.read_csv(f'{path}security.csv', encoding='ISO-8859-1', low_memory=False)
        log_import_security(df_security)

        return df_s12, df_s12type5, df_s34, df_beta, df_security

    except FileNotFoundError:
        return request_wrds(path, start_date, end_date)


def clean_imports(df_s12,
                  df_s12type5,
                  df_s34,
                  df_beta,
                  df_security,
                  start_date,
                  end_date
                  ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_s12_clean = clean_s12(df_s12, start_date, end_date)
    log_clean_s12(df_s12_clean)

    df_s12type5_clean = clean_s12type5(df_s12type5, start_date, end_date)
    log_clean_s12type5(df_s12type5_clean)

    df_s34_clean = clean_s34(df_s34, start_date, end_date)
    log_clean_s34(df_s34_clean)

    df_beta_clean = clean_beta(df_beta, start_date, end_date)
    log_clean_beta(df_beta_clean)

    df_security_clean = clean_security(df_security, start_date, end_date)
    log_clean_security(df_security_clean)

    return df_s12_clean, df_s12type5_clean, df_s34_clean, df_beta_clean, df_security_clean


def clean_s12(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        'fundno',
        'rdate',
        'cusip',
        'shares'
    ]
    return (df[columns]
            .dropna(how='any', subset=['fundno', 'shares', 'cusip'])
            .rename(columns={
        'rdate': 'date',
        'fundno': 'inv_id',
        'cusip': 'asset_id'})
            .assign(date=lambda x: fix_date(x['date']))
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .drop_duplicates(subset=['inv_id', 'date', 'asset_id'], keep='last'))


def clean_s12type5(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    return (df.rename(columns={'fdate': 'date', 'fundno': 'inv_id'})
            .assign(date=lambda x: fix_date(x['date']))
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .dropna(how='any', subset=['inv_id', 'date']))


def clean_s34(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        'mgrno',
        'rdate',
        'typecode',
        'cusip',
        'shares'
    ]
    return (df[columns]
            .dropna(how='any', subset=['mgrno', 'rdate', 'cusip', 'shares'])
            .rename(columns={
        'rdate': 'date',
        'mgrno': 'inv_id',
        'cusip': 'asset_id'})
            .assign(date=lambda x: fix_date(x['date']))
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .drop_duplicates(subset=['inv_id', 'date', 'asset_id'], keep='last'))


def clean_beta(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        'PERMNO',
        'DATE',
        'b_mkt',
        'b_smb',
        'b_hml'
    ]
    offset = 6
    return (df[columns]
            .dropna()
            .rename(columns={
        'DATE': 'date',
        'PERMNO': 'permno'})
            .assign(date=lambda x: fix_date(x['date']) + offset)
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .drop_duplicates(subset=['date', 'permno'], keep='last'))


def clean_security(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        'LPERMNO',
        'cusip',
        'datadate',
        'prccm',
        'trt1m',
        'cshoq'
    ]
    offset = 6
    return (df[columns]
            .rename(columns={
        'LPERMNO': 'permno',
        'cusip': 'asset_id',
        'prccm': 'prc',
        'cshoq': 'shrout',
        'datadate': 'date'})
            .assign(shrout=lambda x: x.groupby('asset_id')['shrout'].ffill() * 1000000,
                    asset_id=lambda x: x['asset_id'].apply(lambda s: s[:-1]),
                    date=lambda x: fix_date(x['date']) + offset)
            .dropna(how='any', subset=['asset_id', 'prc', 'shrout'])
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .drop_duplicates(subset=['date', 'permno'], keep='last'))


# def fix_date(dates: pd.Series) -> pd.Series:
#     return pd.to_datetime(dates) + pd.offsets.MonthEnd(0)


def fix_date(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period(freq='M')


# %%
# Stocks Monthly

def merge_assets_factors(df_assets: pd.DataFrame, df_factors: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.merge(
        left=df_assets,
        right=df_factors,
        how='left',
        on=['date', 'permno'])

    df_merged_indexed = (df_merged
                         .assign(shrout=lambda x: x['shrout'].astype('int64'),
                                 date=lambda x: x['date'].dt.asfreq('Q'))
                         .drop_duplicates(subset=['date', 'asset_id'], keep='last')
                         .drop(columns=['permno']))

    log_asset_merge(df_merged_indexed)
    return df_merged_indexed


# Manager / Holdings

def match_fund_manager(df_fund: pd.DataFrame, df_manager: pd.DataFrame, df_key: pd.DataFrame) -> pd.DataFrame:
    df_fund_manager = (df_manager
                       .assign(shares=lambda x: x['shares'].astype('int64'),
                               date=lambda x: x['date'].dt.asfreq(freq='Q'))
                       .drop_duplicates(subset=['inv_id', 'date', 'asset_id'], keep='last'))

    log_holding_merge(df_fund_manager)
    return df_fund_manager


def construct_zero_holdings(df_fund_manager: pd.DataFrame, n_quarters: int) -> pd.DataFrame:
    def calc_inv_obs(df: pd.DataFrame) -> pd.DataFrame:
        date_diff = (df.groupby('inv_id')['date'].transform('max') - df['date']).apply(lambda x: x.n)
        min_diff = np.minimum(date_diff, n_quarters) + 1
        return min_diff

    def calc_asset_obs(df: pd.DataFrame) -> pd.DataFrame:
        date_diff = (df.sort_values('date')
                     .groupby(['inv_id', 'asset_id'])
                     ['date']
                     .diff(periods=1)
                     .fillna(pd.DateOffset(n=1))
                     .apply(lambda x: x.n))
        min_diff = np.minimum(date_diff, df['inv_obs'])
        return min_diff.apply(lambda x: list(range(x)))

    df_holding = (df_fund_manager
                  .assign(inv_obs=lambda x: calc_inv_obs(x),
                          asset_obs=lambda x: calc_asset_obs(x))
                  .explode('asset_obs')
                  .assign(asset_obs=lambda x: x['asset_obs'].astype('int8'),
                          mask=lambda x: x['asset_obs'] == 0,
                          shares=lambda x: x['shares'] * x['mask'],
                          date=lambda x: x['date'] + x['asset_obs'])
                  .drop(columns=['inv_obs', 'asset_obs', 'mask']))

    log_zero_holdings(df_holding)
    return df_holding


def merge_holding_factor(df_holding: pd.DataFrame, df_asset: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.merge(
        left=df_holding,
        right=df_asset,
        how='left',
        on=['date', 'asset_id'])

    df_holding_factor = (df_merged
                         .assign(shares=lambda x: np.minimum(x['shares'], x['shrout']),
                                 holding=lambda x: x['prc'] * x['shares'] / 1000000,
                                 me=lambda x: x['prc'] * x['shrout'] / 1000000,
                                 typecode=lambda x: x['typecode'].fillna(0).astype('int8'))
                         .dropna(subset='holding'))

    log_holding_factor_merge(df_holding_factor)
    return df_holding_factor


def drop_unsused_inv_asset(df_holding_factor: pd.DataFrame) -> pd.DataFrame:
    df_dropped = (df_holding_factor
                  .assign(Tinv_shares=lambda x: x.groupby(['inv_id', 'date'])['shares'].transform('sum'),
                          Tasset_shares=lambda x: x.groupby(['date', 'asset_id'])['shares'].transform('sum'),
                          mask=lambda x: (x['Tinv_shares'] == 0) | (x['Tasset_shares'] == 0))
                  .loc[lambda x: ~x['mask']]
                  .drop(columns=['Tinv_shares', 'Tasset_shares', 'mask']))

    log_drop_unused(df_dropped)
    return df_dropped


def create_household_sector(df_dropped: pd.DataFrame) -> pd.DataFrame:
    df_household = (df_dropped
                    .groupby(['date', 'asset_id'], as_index=False)
                    .agg({
        'shares': 'sum',
        'prc': 'last',
        'shrout': 'last',
        'b_mkt': 'last',
        'b_smb': 'last',
        'b_hml': 'last',
        'holding': 'sum',
        'me': 'last'})
                    .assign(shares=lambda x: np.maximum(x['shrout'] - x['shares'], 0),
                            holding=lambda x: np.maximum(x['me'] - x['holding'], 0),
                            inv_id=0,
                            typecode=0))

    log_household_sector(df_household)
    df_concat = pd.concat([df_dropped, df_household])
    return df_concat


# %%
# Data

def partition_outside_asset(df_household: pd.DataFrame) -> pd.DataFrame:
    df_outside = (df_household
                  .assign(out_mask=lambda x: x.isna().any(axis=1),
                          out_holding=lambda x: x['holding'] * x['out_mask']))

    log_outside_asset(df_outside)
    return df_outside


def calc_inv_aum(df_outside: pd.DataFrame) -> pd.DataFrame:
    df_grouped = df_outside.groupby(['inv_id', 'date'])

    df_inv_aum = (df_outside
                  .assign(aum=lambda x: df_grouped['holding'].transform('sum'),
                          out_aum=lambda x: df_grouped['out_holding'].transform('sum'),
                          out_weight=lambda x: x['out_aum'] / x['aum']))

    log_inv_aum(df_inv_aum)
    return df_inv_aum


def agg_small_inv(df_inv_aum: pd.DataFrame) -> pd.DataFrame:
    df_agg = (df_inv_aum
              .assign(small_mask=lambda x: (x['aum'] < 10) | (x['out_weight'] == 0) | (x['out_weight'] == 1),
                      inv_id=lambda x: ~x['small_mask'] * x['inv_id'],
                      hh_mask=lambda x: x['inv_id'] == 0))

    df_grouped = df_agg.groupby(['inv_id', 'date', 'asset_id'])

    df_agg = (df_agg
              .assign(Tshares=lambda x: df_grouped['shares'].transform('sum'),
                      Tholding=lambda x: df_grouped['holding'].transform('sum'),
                      shares=lambda x: np.where(x['hh_mask'], x['Tshares'], x['shares']),
                      holding=lambda x: np.where(x['hh_mask'], x['Tholding'], x['holding']))
              .drop(columns=['Tshares', 'Tholding']))

    df_grouped = df_agg.groupby(['inv_id', 'date'])

    df_agg = (df_agg
              .assign(Taum=lambda x: df_grouped['holding'].transform('sum'),
                      Tout_aum=lambda x: df_grouped['out_holding'].transform('sum'),
                      aum=lambda x: np.where(x['hh_mask'], x['Taum'], x['aum']),
                      out_aum=lambda x: np.where(x['hh_mask'], x['Tout_aum'], x['out_aum']),
                      x_holding=lambda x: df_grouped['holding'].transform('count'),
                      n_holding=lambda x: x['x_holding'] - df_grouped['out_mask'].transform('sum'),
                      equal_alloc=lambda x: ~x['hh_mask'] * x['aum'] / (1 + x['x_holding']),
                      drop_mask=lambda x: x['small_mask'] | x['out_mask'])
              .loc[lambda x: ~x['drop_mask']]
              .drop(
        columns=['Taum', 'Tout_aum', 'out_mask', 'out_holding', 'out_weight', 'small_mask', 'drop_mask', 'hh_mask']))

    log_agg_small_inv(df_agg)
    return df_agg


def bin_inv(df_inv_aum: pd.DataFrame, min_n_holding: int) -> (pd.DataFrame, pd.DataFrame):
    def calc_bin(df_type_date: pd.DataFrame) -> pd.Series:
        typecode = df_type_date['typecode'].iloc[0]
        n_holding = (df_type_date['holding'] > 0).sum()
        n_bins = np.floor(n_holding / (2 * min_n_holding)).astype(int)
        idx = df_type_date.index

        if (n_bins <= 1) | (n_holding >= min_n_holding) | (typecode == 0):
            return pd.Series(0, index=idx)
        else:
            return pd.Series(pd.qcut(x=df_type_date['aum'], q=n_bins, labels=False), index=idx)

    df_binned = (df_inv_aum
                 .reset_index(drop=True)
                 .assign(bin=lambda x: x.groupby(['typecode', 'date']).apply(calc_bin).reset_index(drop=True))
                 .assign(bin=lambda x: x.groupby(['date', 'typecode', 'bin']).ngroup()))

    log_bins(df_binned)
    return df_binned


def calc_instrument(df_binned: pd.DataFrame) -> pd.DataFrame:
    df_instrument = (df_binned
                     .assign(total_alloc=lambda x: x.groupby(['date', 'asset_id'])['equal_alloc'].transform('sum'),
                             iv_me=lambda x: x['total_alloc'] - x['equal_alloc'])
                     .drop(columns=['total_alloc', 'equal_alloc']))

    log_instrument(df_instrument)
    return df_instrument


def dask_calc_instrument(df_binned: pd.DataFrame) -> pd.DataFrame:
    df_binned = dd.from_pandas(df_binned.set_index('date'), npartitions=32)

    df_instrument = (df_binned
                     .assign(total_alloc=lambda x: x.groupby(['date', 'asset_id'])['equal_alloc'].transform('sum',
                                                                                                            meta=pd.Series(
                                                                                                                dtype='float64')),
                             iv_me=lambda x: x['total_alloc'] - x['equal_alloc'])
                     .drop(columns=['total_alloc', 'equal_alloc'])
                     .compute()
                     .reset_index())

    log_instrument(df_instrument)
    return df_instrument


def calc_holding_weights(df_instrument: pd.DataFrame) -> pd.DataFrame:
    mask = (df_instrument['iv_me'] > 0)

    df_weights = (df_instrument.loc[mask]
                  .assign(ln_me=lambda x: np.log(x['me']),
                          ln_iv_me=lambda x: np.log(x['iv_me']),
                          weight=lambda x: x['holding'] / x['aum'],
                          rweight=lambda x: x['holding'] / x['out_aum'],
                          ln_rweight=lambda x: np.log(x['rweight'].mask(x['rweight'] == 0, np.NaN)),
                          mean_ln_rweight=lambda x: x.groupby(['inv_id', 'date'])['ln_rweight'].transform('mean'),
                          const=1,
                          pct_uni_held=lambda x: np.log(x['n_holding']) - np.log(x['x_holding']))
                  .drop(columns=['me', 'iv_me']))

    log_holding_weights(df_weights)
    return df_weights


# %%
# Estimation

def bound_float(arr: np.array) -> float:
    ln_bound = 709.7827
    return np.minimum(np.maximum(arr, -ln_bound), ln_bound)


def momcond_L(params: np.array, exog: np.array) -> np.array:
    upper_bound = 0.999
    exog = exog.T

    beta_ln_me = params[0]
    beta_characteristics = params[1:]

    ln_me = exog[0]
    mean_ln_rweight = exog[1]
    arr_characteristics = exog[2:]

    ln_me_term = (upper_bound - np.exp(-1 * beta_ln_me)) * ln_me
    characteristics_term = np.dot(beta_characteristics, arr_characteristics)
    pred_weight = ln_me_term + characteristics_term + mean_ln_rweight

    return pred_weight


def fit_inv_date_L(df_inv_date: pd.DataFrame, characteristics: list, params: list, ) -> gmm.GMMResults:
    df_inv_date = df_inv_date.dropna(subset='ln_rweight')

    exog = np.asarray(df_inv_date[['ln_me', 'mean_ln_rweight'] + characteristics])
    instrument = np.asarray(df_inv_date[['ln_iv_me', 'mean_ln_rweight'] + characteristics])
    n = exog.shape[0]
    endog = np.asarray(df_inv_date['ln_rweight'] - df_inv_date['pct_uni_held'])
    start_params = np.zeros(len(params))
    w0inv = np.dot(instrument.T, instrument) / n

    try:
        model = gmm.NonlinearIVGMM(
            endog=endog,
            exog=exog,
            instrument=instrument,
            func=momcond_L)
        result = model.fit(
            start_params=start_params,
            maxiter=0,
            inv_weights=w0inv)
        # log_results(result, params)
        return result

    except np.linalg.LinAlgError:
        print('Linear Algebra Error')
        return None


def estimate_model_L(df_weights: pd.DataFrame, characteristics: list, params: list, min_n_holding: int) -> pd.DataFrame:
    mask = df_weights['n_holding'] >= min_n_holding

    df_institutions = (df_weights
                       .loc[mask]
                       .set_index(['inv_id', 'date'])
                       .assign(
        gmm_result=lambda x: x.groupby(['inv_id', 'date']).apply(lambda y: fit_inv_date_L(y, characteristics, params)))
                       .reset_index())

    df_bins = (df_weights
               .loc[~mask]
               .set_index(['bin', 'date'])
               .assign(
        gmm_result=lambda x: x.groupby(['bin', 'date']).apply(lambda y: fit_inv_date_L(y, characteristics, params)))
               .reset_index())

    df_model = pd.concat([df_institutions, df_bins])
    log_params(df_model)
    return df_model


def momcond_NL(params: np.array, exog: np.array) -> np.array:
    upper_bound = 0.9999
    exog = exog.T

    beta_ln_me = bound_float(params[0])
    beta_characteristics = params[1:]

    ln_me = exog[0]
    rweight = exog[1]
    mean_ln_rweight = exog[2]
    arr_characteristics = exog[3:]

    ln_me_term = (upper_bound - np.exp(-1 * beta_ln_me)) * ln_me
    characteristics_term = np.dot(beta_characteristics, arr_characteristics)
    pred_ln_rweight = bound_float(ln_me_term + characteristics_term + mean_ln_rweight)
    pred_rweight = np.exp(-1 * pred_ln_rweight)

    return rweight * pred_rweight


def fit_inv_date_NL(df_inv_date: pd.DataFrame, characteristics: list, params: list) -> gmm.GMMResults:
    exog = np.asarray(df_inv_date[['ln_me', 'rweight', 'mean_ln_rweight'] + characteristics])
    instrument = np.asarray(df_inv_date[['ln_iv_me', 'rweight', 'mean_ln_rweight'] + characteristics])
    n = exog.shape[0]
    endog = np.ones(n)

    try:
        model = gmm.NonlinearIVGMM(
            endog=endog,
            exog=exog,
            instrument=instrument,
            func=momcond_NL)
        w0inv = np.dot(instrument.T, instrument) / n
        start_params = np.zeros(len(params))
        result = model.fit(
            start_params=start_params,
            maxiter=100,
            inv_weights=w0inv)
        # log_results(result, params)
        return result

    except np.linalg.LinAlgError:
        print('Linear Algebra Error')
        return None


def estimate_model_NL(df_weights: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    mask = df_weights['n_holding'] >= min_n_holding

    df_institutions = (df_weights
                       .loc[mask]
                       .set_index(['inv_id', 'date'])
                       .assign(
        gmm_result=lambda x: x.groupby(['inv_id', 'date']).apply(lambda y: fit_inv_date_L(y, characteristics, params)))
                       .reset_index())

    df_bins = (df_weights
               .loc[~mask]
               .set_index(['bin', 'date'])
               .assign(
        gmm_result=lambda x: x.groupby(['bin', 'date']).apply(lambda y: fit_inv_date_L(y, characteristics, params)))
               .reset_index())

    df_model = pd.concat([df_institutions, df_bins])
    log_params(df_model)
    return df_model


def calc_latent_demand_L(df_model: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    def unpack_result(result: gmm.GMMResults) -> list:
        return result.params

    def unpack_params(df: pd.DataFrame) -> pd.DataFrame:
        df[params] = pd.DataFrame(df['lst_params'].tolist(), index=df.index)
        return df

    upper_bound = 0.9999

    df_results = (df_model
                  .dropna()
                  .assign(lst_params=lambda x: x['gmm_result'].apply(unpack_result))
                  .pipe(unpack_params)
                  .drop(columns='lst_params')
                  .assign(beta_ln_me=lambda x: upper_bound - np.exp(-1 * x['beta_ln_me']),
                          beta_const=lambda x: x['beta_const'] + x['mean_ln_rweight'],
                          pred_ln_rweight=lambda x: np.einsum('ij,ij->i', x[['ln_me'] + characteristics], x[params]),
                          latent_demand=lambda x: x['ln_rweight'] - x['pred_ln_rweight'])
                  .drop(columns=['pct_uni_held', 'gmm_result', 'pred_ln_rweight']))

    log_latent_demand(df_results)
    return df_results


# %%
# Liquidity

def calc_coliquidity_matrix(df_date: pd.DataFrame) -> pd.Series:
    df_asset_grouped = df_date.groupby('asset_id')

    n_assets = len(df_asset_grouped)

    df_liquid = (df_date
                 .assign(first=lambda x: df_asset_grouped.cumcount() == 0,
                         total_holding=lambda x: df_asset_grouped['holding'].transform('sum'),
                         share_of_holding=lambda x: x['holding'] / x['total_holding'],
                         row=lambda x: df_asset_grouped.ngroup())
                 .sort_values('row'))

    Zmat = np.empty(shape=(n_assets, n_assets))
    Amat = np.empty(shape=(n_assets, n_assets))

    for col in range(n_assets):
        df_liquid = df_liquid.assign(mask=lambda x: x['row'] == col,
                                     mask_weight=lambda x: x['mask'] * x['weight'],
                                     rcweight=lambda x: x['mask'] - x.groupby('inv_id')['mask_weight'].transform('max'),
                                     Zcol=lambda x: x['beta_ln_me'] * x['share_of_holding'] * x['rcweight'],
                                     Acol=lambda x: x['share_of_holding'] * x['rcweight'])
        Zmat[col] = df_liquid.groupby('asset_id')['Zcol'].sum()
        Amat[col] = df_liquid.groupby('asset_id')['Acol'].sum()

    # types = df_date['typecode'].unique()
    # for i, t in enumerate(types):
    #     Bcol_name = 'Bcol_' + str(t)
    #     mask = df_liquid['typecode'] == t
    #     df_liquid[Bcol_name] = mask * df_liquid['Acol']
    #
    # for t in types:
    #     Bcol_name = 'Bcol_' + str(t)
    #
    #     n_type = df_asset_grouped.apply(lambda x: (x['typecode'] == t).sum())
    #     df_matrix[Bcol_name] = df_asset_grouped[Bcol_name].sum() / np.maximum(n_type, 1)

    I = np.identity(n_assets)
    liquidity_matrixA = np.linalg.solve(I - Zmat, Amat)
    price_impact = np.diagonal(liquidity_matrixA)

    idx = df_liquid.set_index('row')['asset_id'].to_dict()
    sr_liquid = pd.Series(price_impact, index=idx, name='price_impact')

    return sr_liquid


def calc_price_elasticity(df_results: pd.DataFrame) -> pd.DataFrame:
    mask = df_results['holding'] > 0

    df_liquidity = (df_results
                    .loc[mask]
                    .groupby('date')
                    .apply(calc_coliquidity_matrix))

    log_liquidity(df_liquidity)
    return df_liquidity


def graph_liquidity(df_liquidity: pd.DataFrame):
    df_fig = (df_liquidity
              .groupby('date')
              .agg({'p10': lambda x: np.quantile(x['price_impact'], q=0.1),
                    'p50': lambda x: np.quantile(x['price_impact'], q=0.5),
                    'p90': lambda x: np.quantile(x['price_impact'], q=0.9)})
              .stack()
              .reset_index(name='stat'))

    g = sns.relplot(data=df_fig,
                    x='date',
                    y='p10',
                    hue='stat',
                    kind='line')
    g.set_axis_labels('Date', 'Price Impact')
    g.legend()
    g.despine()
    plt.savefig(os.path.join(figure_path, f'price_elasticity.png'))
    return df_liquidity


# %%
# Decomposition

def equillibrium_loop(df_loop: pd.DataFrame) -> pd.DataFrame:
    for i in range(1000):
        df_loop = (df_loop
                   .assign(weightN=lambda x: np.exp(x['loop_beta_ln_me'] * x['loop_ln_prc']) + x['loop_latent'],
                           weightD=lambda x: x.groupby(['inv_id', 'date'])['weightN'].transform('sum'),
                           weight=lambda x: x['weightN'] / (1 + x['weightD']),
                           holding=lambda x: x['weight'] * x['loop_aum'],
                           demand=lambda x: x.groupby(['inv_id', 'date'])['holding'].transform('sum'),
                           Ddemand_a=lambda x: x['loop_beta_ln_me'] * x['loop_aum'] * x['weight'] * (1 - x['weight']) /
                                               x['demand'],
                           Ddemand_b=lambda x: x.groupby(['date', 'asset_id'])['Ddemand_a'].transform('sum'),
                           Ddemand_c=lambda x: 1 / (1 - np.minimum(0, x['Ddemand_b'])),
                           gap=lambda x: np.log(x['demand']) - x['loop_ln_price'] - x['loop_ln_shrout']))

        gap = max(abs(df_loop['gap'].min()), abs(df_loop['gap'].max()))

        if i == 999:
            print(f'Equilibrium did not converge after 1000 iterations')
            print(f'Gap:  ', gap)

        if gap < 0.00001:
            print(f'Equilibrium converged after {i} iterations')
            df_loop = df_loop.assign(loop_ln_prc=lambda x: x.groupby('asset_id')['loop_ln_prc'].transform('min'))

    return df_loop


def run_counterfactuals(df_date: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    for i in range(1, 6):
        if i <= 4:
            df_date = df_date.assign(loop_ln_prc=lambda x: x['lag_ln_prc'],
                                     loop_ln_shrout=lambda x: x['lag_ln_shrout'])
        else:
            df_date = df_date.assign(loop_ln_prc=lambda x: x['prc'],
                                     loop_ln_shrout=lambda x: x['shrout'])

        if i == 1:
            for char in characteristics:
                df_date['loop_' + char] = df_date['lag_' + char]
        elif i <= 4:
            for char in characteristics:
                df_date['loop_' + char] = df_date['lag_' + char]
        else:
            for char in characteristics:
                df_date['loop_' + char] = df_date[char]

        if i <= 2:
            df_date['loop_aum'] = df_date['lag_aum']
        elif i <= 4:
            df_date['loop_aum'] = df_date['aum']
        else:
            df_date['loop_aum'] = df_date['aum']

        if i <= 3:
            for param in params:
                df_date['loop_' + param] = df_date['lag_' + param]
        elif i <= 4:
            for param in params:
                df_date['loop_' + param] = df_date['lag_' + param]
        else:
            for param in params:
                df_date['loop_' + param] = df_date[param]

        loop_chars = ['loop_' + char for char in characteristics]
        loop_params = ['loop_' + param for param in params]
        if i <= 4:
            df_date = df_date.assign(
                loop_latent=np.einsum('ij,ij->i', df_date[['loop_ln_shrout'] + loop_chars], df_date[loop_params]) + x[
                    'lag_latent_demand'])
        else:
            df_date = df_date.assign(
                loop_latent=np.einsum('ij,ij->i', df_date[['loop_ln_shrout'] + loop_chars], df_date[loop_params]) + x[
                    'latent_demand'])

    return df_date


def decompose_variance(df_results: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    mask = df_results['holding'] > 0

    df_variance = (df_results
                   .loc[mask]
                   .assign(ln_prc=lambda x: np.log(x['prc']),
                           ln_shrout=lambda x: np.log(x['shrout'])))

    df_grouped = df_variance.groupby(['date', 'asset_id'])

    df_variance = (df_variance
                   .assign(lag_ln_prc=lambda x: df_grouped['ln_prc'].shift(4),
                           lag_ln_shrout=lambda x: df_grouped['ln_shrout'].shift(4),
                           lag_aum=lambda x: df_grouped['aum'].shift(4),
                           lag_latent=lambda x: df_grouped['latent_demand'].shift(4), )

                   for char in characteristics:
    name = 'lag_' + char
    df_variance[name] = df_variance.groupby(['date', 'asset_id'])[char].shift(4)

    return df_variance


# %%
# Predictability


# %%
# Figures

def clean_figures(df_results: pd.DataFrame) -> pd.DataFrame:
    return (df_results
            .assign(date=lambda x: x['date'].dt.to_timestamp(),
                    typecode=lambda x: x['typecode'].apply(get_readable_typecode)))


def typecode_share_counts(df_figure: pd.DataFrame, figure_path: str):
    df_type_share = (df_figure
                     .groupby(['typecode', 'date'])
                     .agg({'aum': 'sum', 'n_holding': 'count'})
                     .assign(share=lambda x: x['aum'] / x['aum'].groupby('date').transform('sum')))

    g = sns.relplot(data=df_type_share,
                    x='date',
                    y='n_holding',
                    hue='typecode',
                    kind='line')
    g.set_axis_labels('Date', 'Number of Investors')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'typecode_count.png'))

    g = sns.relplot(data=df_type_share,
                    x='date',
                    y='share',
                    hue='typecode',
                    kind='line')
    g.set_axis_labels('Date', 'Share of AUM')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'typecode_share.png'))

    return df_inv_aum


def check_moment_condition(df_figures: pd.DataFrame, min_n_holding: int):
    mask = df_figures['n_holding'] >= min_n_holding
    df_mom = (df_figures
              .loc[mask]
              .groupby(['inv_id', 'date'])
              .agg({'latent_demand': 'mean'}))

    epsilon = 0.01
    mask = (df_mom['latent_demand'] > -epsilon) & (df_mom['latent_demand'] < epsilon)
    valid_rate = len(df_mom[mask]) / len(df_mom)
    print(f'Percentage of valid moments:  {100 * valid_rate:.4f}%')

    g = sns.displot(
        data=df_mom,
        x='latent_demand'
    )
    g.set_axis_labels('Log Latent Demand', 'Frequency')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'moment_condition.png'))

    return df_figures


def critical_value_test(df_figures: pd.DataFrame, characteristics: list, min_n_holding: int, figure_path: str):
    def iv_reg(df_inv_date: pd.DataFrame):
        y = df_inv_date['ln_me']
        X = df_inv_date[['ln_iv_me'] + characteristics]
        model = OLS(y, X)
        result = model.fit()
        t_stat = result.tvalues.iloc[0]

        return t_stat

    mask = df_figures['n_holding'] >= min_n_holding

    df_iv_a = (df_figures
               .loc[mask]
               .groupby(['inv_id', 'date'])
               .apply(iv_reg))

    df_iv_b = (df_figures
               .loc[~mask]
               .groupby(['bin', 'date'])
               .apply(iv_reg))

    df_iv = (pd.concat([df_iv_a, df_iv_b])
             .to_frame('t_stat')
             .groupby('date')
             .min())

    g = sns.relplot(data=df_iv,
                    x='date',
                    y='t_stat',
                    kind='line')
    g.refline(y=4.05, linestyle='--')
    g.set_axis_labels('Date', 'First stage t-statistic')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'instrument_validity.png'))

    return df_figures


def test_index_fund(df_figures: pd.DataFrame, characteristics: list, params: list, indexfund_id: int, figure_path: str):
    mask = df_figures['inv_id'] == indexfund_id
    df_index_fund = (df_figures
    .assign(ln_rweight=lambda x: x['ln_me'] + x['mean_ln_rweight'],
            pct_uni_held=1)
    .loc[mask])

    df_index_fund_model = (df_index_fund
                           .set_index(['inv_id', 'date'])
                           .assign(
        gmm_result=lambda x: x.groupby(['inv_id', 'date']).apply(lambda y: fit_inv_date_L(y, characteristics, params)))
                           .reset_index())
    df_index_fund_result = calc_latent_demand_L(df_index_fund_model, characteristics, params)
    cols = params + ['latent_demand']
    for param in cols:
        g = sns.relplot(data=df_index_fund_result,
                        x='date',
                        y=param,
                        kind='line')
        g.set_axis_labels('Date', f'{get_readable_param(param)}')
        g.despine()
        plt.ylim(-1, 1)
        plt.savefig(os.path.join(figure_path, f'index_fund_{param}.png'))

    return df_figures


def graph_type_params(df_results: pd.DataFrame, params: list, figure_path: str):
    df_types = df_results.copy()
    df_types[params] = df_types[params].apply(lambda x: x * df_results['aum'])
    df_types = (df_types
                .groupby(['typecode', 'date'])
                [params]
                .sum()
                .apply(lambda x: x / df_results.groupby(['typecode', 'date'])['aum'].sum()))

    for param in params:
        g = sns.relplot(data=df_types,
                        x='date',
                        y=param,
                        hue='typecode',
                        kind='line')
        g.set_axis_labels('Date', f'{get_readable_param(param)}')
        g.legend.set_title('Institution Type')
        g.despine()
        plt.savefig(os.path.join(figure_path, f'{param}.png'))

    return df_results


def graph_std_latent_demand(df_figures: pd.DataFrame, figure_path: str):
    df_ld = (df_figures
             .groupby(['inv_id', 'date'])
             .agg({
        'latent_demand': 'std',
        'aum': 'last',
        'typecode': 'last'})
             .assign(latent_demand=lambda x: x['latent_demand'] * x['aum'])
             .groupby(['typecode', 'date'])
             .agg({
        'latent_demand': 'mean',
        'aum': 'sum'})
             .assign(latent_demand=lambda x: x['latent_demand'] / x['aum']))

    g = sns.relplot(data=df_ld,
                    x='date',
                    y='latent_demand',
                    hue='typecode',
                    kind='line')
    g.set_axis_labels('Date', 'Standard Deviation of Latent Demand')
    g.legend.set_title('Institution Type')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'std_latent_demand.png'))

    return df_results


def summary_tables(df_figures: pd.DataFrame):
    return df_figures


def get_param_cols(cols: list) -> list:
    return ['beta_' + col for col in cols]


def get_readable_param(name: str) -> str:
    return name.replace('_', ' ').title()


def get_readable_typecode(typecode: int):
    return dict_typecode[typecode]


# %%
# Log

def log_import_s12(df_s12: pd.DataFrame):
    dateindex_s12 = df_s12['rdate']
    print('Imported s12')
    print('Number of holdings:  ', len(df_s12))
    print('Earliest date:  ', min(dateindex_s12))
    print('Latest date:  ', max(dateindex_s12))


def log_import_s12type5(df_s12type5: pd.DataFrame):
    dateindex_s12type5 = df_s12type5['fdate']
    print()
    print('Imported s12type5')
    print('Number of holdings:  ', len(df_s12type5))
    print('Earliest date:  ', min(dateindex_s12type5))
    print('Latest date:  ', max(dateindex_s12type5))


def log_import_s34(df_s34: pd.DataFrame):
    dateindex_s34 = df_s34['rdate']
    print()
    print('Imported s34')
    print('Number of holdings:  ', len(df_s34))
    print('Earliest date:  ', min(dateindex_s34))
    print('Latest date:  ', max(dateindex_s34))


def log_import_beta(df_beta: pd.DataFrame):
    dateindex_ffm = df_beta['DATE']
    print()
    print('Imported betas')
    print('Number of dates:  ', len(df_beta))
    print('Earliest date:  ', min(dateindex_ffm))
    print('Latest date:  ', max(dateindex_ffm))


def log_import_security(df_security: pd.DataFrame):
    dateindex_security = df_security['datadate']
    print()
    print('Imported security')
    print('Number of holdings:  ', len(df_security))
    print('Earliest date:  ', min(dateindex_security))
    print('Latest date:  ', max(dateindex_security))


def log_clean_s12(df_s12_clean: pd.DataFrame):
    dateindex_s12 = df_s12_clean['date']
    print()
    print('Cleaned s12')
    print('Number of holdings:  ', len(df_s12_clean))
    print('Earliest date:  ', min(dateindex_s12))
    print('Latest date:  ', max(dateindex_s12))


def log_clean_s12type5(df_s12type5_clean: pd.DataFrame):
    dateindex_s12type5 = df_s12type5_clean['date']
    print()
    print('Cleaned s12type5')
    print('Number of firm/dates:  ', len(df_s12type5_clean))
    print('Earliest date:  ', min(dateindex_s12type5))
    print('Latest date:  ', max(dateindex_s12type5))


def log_clean_s34(df_s34_clean: pd.DataFrame):
    dateindex_s34 = df_s34_clean['date']
    print()
    print('Cleaned s34')
    print('Number of holdings:  ', len(df_s34_clean))
    print('Earliest date:  ', min(dateindex_s34))
    print('Latest date:  ', max(dateindex_s34))


def log_clean_beta(df_beta_clean: pd.DataFrame):
    dateindex_beta = df_beta_clean['date']
    print()
    print('Cleaned beta')
    print('Number of dates:  ', len(df_beta_clean))
    print('Earliest date:  ', min(dateindex_beta))
    print('Latest date:  ', max(dateindex_beta))


def log_clean_security(df_security_clean: pd.DataFrame):
    dateindex_security = df_security_clean['date']
    print()
    print('Cleaned security')
    print('Number of asset/dates:  ', len(df_security_clean))
    print('Earliest date:  ', min(dateindex_security))
    print('Latest date:  ', max(dateindex_security))


def log_holding_merge(df_merged: pd.DataFrame):
    print()
    print('Merged s12 and s34')
    print('Number of holdings:  ', len(df_merged))
    print()


def log_drop_unused(df_dropped: pd.DataFrame):
    print()
    print('Dropped unused investors/assets')
    print('Number of holdings:  ', len(df_dropped))
    print()


def log_asset_merge(df_merged: pd.DataFrame):
    print()
    print('Merged assets and factors')
    print('Number of assets/dates:  ', len(df_merged))
    print()


def log_holding_factor_merge(df_merged: pd.DataFrame):
    print()
    print('Merged holdings and factors')
    print('Number of investor/date/asset:  ', len(df_merged))
    print()


def log_household_sector(df_household: pd.DataFrame):
    print()
    print('Created household sector')
    print('Number of holdings:  ', len(df_household))


def log_outside_asset(df_outside: pd.DataFrame):
    print()
    print('Created outside asset')
    print('Number of outside holdings:  ', df_outside['out_mask'].sum())


def log_zero_holdings(df_holding: pd.DataFrame):
    print()
    print('Constructed zero holdings')
    print('Number of non-zero holdings:  ', sum(df_holding['shares'] > 0))
    print('Number of zero holdings:  ', sum(df_holding['shares'] == 0))


def log_inv_aum(df_inv_aum: pd.DataFrame):
    print()
    print('Calculated investor AUM')
    print('Number of investors:  ', df_inv_aum['inv_id'].nunique())


def log_agg_small_inv(df_agg: pd.DataFrame):
    print()
    print('Aggregated small investors')
    print('Number of investors:  ', df_agg['inv_id'].nunique())


def log_bins(df_binned: pd.DataFrame):
    print()
    print('Binned investors')
    print('Number of investors:  ', df_binned['inv_id'].nunique())
    print('Number of bins:  ', df_binned['bin'].nunique())


def log_inv_universe(df_inv_uni: pd.DataFrame):
    print()
    print('Created investment universe')
    print('Number of investors:  ', df_inv_uni['inv_id'].nunique())
    print('Average investment universe size:  ', df_inv_uni['uni_size'].mean())


def log_instrument(df_instrument: pd.DataFrame):
    print()
    print('Created market equity instrument')
    print('Number of valid instruments:  ', sum(df_instrument['iv_me'] > 0))
    print('Number of invalid instruments:  ', sum(df_instrument['iv_me'] == 0))


def log_holding_weights(df_model: pd.DataFrame):
    print()
    print('Calculated holding weights')
    print('Number of non-zero holdings:  ', sum(df_model['holding'] > 0))
    print('Number of zero holdings:  ', sum(df_model['holding'] == 0))


def log_results(result, params):
    print()
    print(result.summary(yname='Latent demand', xname=params))
    print()


def log_params(df_params: pd.DataFrame):
    print()
    print('Estimated parameters')
    print('Number of converged estimations:  ', sum(df_params['gmm_result'].notna()))
    print('Number of unconverged estimations:  ', sum(df_params['gmm_result'].isna()))


def log_latent_demand(df_results: pd.DataFrame):
    print()
    print('Calculated latent demand')
    print('Number of investors:  ', df_results['inv_id'].nunique())


def log_liquidity(df_liquidity: pd.DataFrame):
    print()
    print('Calculated price elasticity')


def log_variance(df_variance: pd.DataFrame):
    print()
    print('Decomposed variance')


def log_predictability(df_predictability: pd.DataFrame):
    print()
    print('Tested return predictability')


# %%
def main():
    client = Client()

    dfs = pandas_read(input_path)
    df_s12_clean, df_s12type5_clean, df_s34_clean, df_beta_clean, df_security_clean = clean_imports(
        *dfs,
        start_date,
        end_date
    )

    df_fund_manager = match_fund_manager(df_s12_clean, df_s34_clean, df_s12type5_clean)

    df_holding = (df_fund_manager
                  .pipe(construct_zero_holdings, n_quarters))

    df_asset = merge_assets_factors(df_security_clean, df_beta_clean)
    df_holding_factor = merge_holding_factor(df_holding, df_asset)

    df_results = (df_holding_factor
                  .pipe(drop_unsused_inv_asset)
                  .pipe(create_household_sector)
                  .pipe(partition_outside_asset)
                  .pipe(calc_inv_aum)
                  .pipe(agg_small_inv)
                  .pipe(bin_inv, min_n_holding)
                  .pipe(calc_instrument)
                  .pipe(calc_holding_weights)
                  .pipe(estimate_model_L, characteristics, params, min_n_holding)
                  .pipe(calc_latent_demand_L, characteristics, params))
    return df_results
    df_results.to_csv(os.path.join(output_path, 'df_results.csv'))

    df_figures = (df_results
                  .pipe(clean_figures)
                  .pipe(typecode_share_counts, figure_path)
                  .pipe(check_moment_condition, min_n_holding)
                  .pipe(critical_value_test, characteristics, figure_path)
                  .pipe(test_index_fund, characteristics, params, figure_path)
                  .pipe(graph_type_params, params, figure_path)
                  .pipe(graph_std_latent_demand, figure_path))


# %%
# Global parameters

sns.set_theme(style='ticks', palette=sns.color_palette('hls', 6), context='paper')

input_path = 'data/'
output_path = 'output/'
figure_path = 'figures/'

os.makedirs(output_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

start_date = pd.Period('2012-01')
end_date = pd.Period('2017-12')

characteristics = [
                      'b_mkt',
                      'b_smb',
                      'b_hml'
                  ] + ['const']
params = ['beta_ln_me'] + get_param_cols(characteristics)
dict_typecode = {
    0: 'Households',
    1: 'Banks',
    2: 'Insurance companies',
    3: 'Investment advisors',
    4: 'Mutual funds',
    5: 'Pension funds',
}

min_n_holding = 1000
n_quarters = 11
indexfund_id = 90457
# %%
df_result = main()
# %%
df_liquidity = calc_price_elasticity(df_result)
# %%
df_liquidity.describe()
# %%
print('\n---------------Starting Imports---------------------------\n')
dfs = pandas_read(input_path)
# %%
print('\n---------------Starting Cleaning---------------------------\n')
df_s12_clean, df_s12type5_clean, df_s34_clean, df_beta_clean, df_security_clean = clean_imports(
    *dfs,
    start_date,
    end_date
)

# df_s12_clean.to_csv(os.path.join(output_path, 'df_s12_clean.csv'))
# df_s12type5_clean.to_csv(os.path.join(output_path, 'df_s12type5_clean.csv'))
# df_s34_clean.to_csv(os.path.join(output_path, 'df_s34_clean.csv'))
# df_beta_clean.to_csv(os.path.join(output_path, 'df_beta_clean.csv'))
# df_security_clean.to_csv(os.path.join(output_path, 'df_security_clean.csv'))
# %%
print('\n---------------Merging Assets/Factors---------------------------\n')
df_asset = merge_assets_factors(df_security_clean, df_beta_clean)
# df_asset.to_csv(os.path.join(output_path, 'df_asset.csv'))

print('\n---------------Merging s12/s34 Holdings---------------------------\n')
df_fund_manager = match_fund_manager(df_s12_clean, df_s34_clean, df_s12type5_clean)
# df_fund_manager.to_csv(os.path.join(output_path, 'df_fund_manager.csv'))
# %%
print('\n---------------Constructing Zero Holdings---------------------------\n')
df_holding = construct_zero_holdings(df_fund_manager, n_quarters)
# df_holding.to_csv(os.path.join(output_path, 'df_holding.csv'))
# %%
print('\n---------------Merging Holdings/Factors---------------------------\n')
df_holding_factor = merge_holding_factor(df_holding, df_asset)
# df_holding_factor.to_csv(os.path.join(output_path, 'df_holding_factor.csv'))
# %%
print('\n---------------Dropping Mangers/Assets Without Holdings---------------------------\n')
df_dropped = drop_unsused_inv_asset(df_holding_factor)
# df_holding.to_csv(os.path.join(output_path, 'df_holding.csv'))

print('\n---------------Creating Household Sector---------------------------\n')
df_household = create_household_sector(df_dropped)
# df_household.to_csv(os.path.join(output_path, 'df_household.csv'))

print('\n---------------Partitioning Outside Asset---------------------------\n')
df_outside = partition_outside_asset(df_household)
# df_outside.to_csv(os.path.join(output_path, 'df_outside.csv'))
# %%
print('\n---------------Calculating Investor AUM---------------------------\n')
df_inv_aum = calc_inv_aum(df_outside)
# df_inv_aum.to_csv(os.path.join(output_path, 'df_inv_aum.csv'))
# %%
print('\n---------------Aggregating Small Investors---------------------------\n')
df_agg = agg_small_inv(df_inv_aum)
# df_agg.to_csv(os.path.join(output_path, 'df_agg.csv'))
# %%
print('\n---------------Pooling Investors By Type/Size---------------------------\n')
df_binned = bin_inv(df_agg, min_n_holding)
# df_binned.to_csv(os.path.join(output_path, 'df_binned.csv'))
# %%
print('\n---------------Calculating Instrument---------------------------\n')
df_instrument = calc_instrument(df_binned)
# df_instrument.to_csv(os.path.join(output_path, 'df_instrument.csv'))
# %%
print('\n---------------Calculating Holding Weights---------------------------\n')
df_weights = calc_holding_weights(df_instrument)
# df_weights.to_csv(os.path.join(output_path, 'df_weights.csv'))
# %%
print('\n---------------Estimating Demand System---------------------------\n')
df_model = estimate_model_L(df_weights, characteristics, params, min_n_holding)
# df_model.to_csv(os.path.join(output_path, 'df_model.csv'))
# %%
print('\n---------------Calculating Latent Demand---------------------------\n')
df_results = calc_latent_demand_L(df_model, characteristics, params)
# df_results.to_csv(os.path.join(output_path, 'df_results.csv'))
# %%
print('\n---------------Testing Moment Condition---------------------------\n')
# df_figures = clean_figures(df_results)
# _ = check_moment_condition(df_figures, 0, min_n_holding)
# %%
print('\n---------------Investors by Typecode---------------------------\n')
# _ = typecode_share_counts(df_figures, figure_path)
# %%
print('\n---------------Testing Instrument Validity---------------------------\n')
# _ = critical_value_test(df_figures, characteristics, min_n_holding, figure_path)
# %%
print('\n---------------Testing Hypothetical Index Fund---------------------------\n')
# _ = test_index_fund(df_figures, characteristics, params, indexfund_id, figure_path)
# %%
print('\n---------------Parameters By Typecode---------------------------\n')
# _ = graph_type_params(df_figures, params, figure_path)
# %%
print('\n---------------Standard Deviation of Latent Demand By Typecode---------------------------\n')
# _ = graph_std_latent_demand(df_figures, figure_path)
# %%
print('\n---------------Calculating Price Elasticity---------------------------\n')
df_liquidity = calc_price_elasticity(df_results).pipe(graph_liquidity)
# %%
print('\n---------------Testing Predictability---------------------------\n')

# %%
print('\n---------------Simulating Monetary Policy Shock---------------------------\n')

# %%
print('\n---------------Finished---------------------------\n')