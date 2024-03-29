# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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


def load_wrds(path: str,
              start_date: pd.Timestamp,
              end_date: pd.Timestamp
              ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
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
            .drop_duplicates(subset=['inv_id', 'date', 'asset_id'], keep='last')
            .set_index(['inv_id', 'date', 'asset_id']))


def clean_s12type5(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    return (df
            .rename(columns={'fdate': 'date', 'fundno': 'inv_id'})
            .assign(
        date=lambda x: fix_date(x['date']))
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .dropna(how='any', subset=['inv_id', 'date'])
            .set_index(['inv_id', 'date']))


def clean_s34(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        'mgrno',
        'rdate',
        'typecode',
        'cusip',
        'prc',
        'shrout2',
        'shares'
    ]
    return (df[columns]
            .dropna(how='any', subset=['cusip', 'shares'])
            .rename(columns={
        'rdate': 'date',
        'mgrno': 'inv_id',
        'cusip': 'asset_id'})
            .assign(
        date=lambda x: fix_date(x['date']),
        backup_holding=lambda x: x['shares'] * x['prc'],
        backup_me=lambda x: x['shrout2'] * x['prc'] * 1000,
        typecode=lambda x: x['typecode'].fillna(0).astype(int))
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .drop_duplicates(subset=['inv_id', 'date', 'asset_id'], keep='last')
            .drop(columns=['prc', 'shrout2'])
            .set_index(['inv_id', 'date', 'asset_id']))


def clean_beta(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        'PERMNO',
        'DATE',
        'b_mkt',
        'b_smb',
        'b_hml'
    ]
    offset = pd.DateOffset(months=6)
    return (df[columns]
            .dropna()
            .rename(columns={
        'DATE': 'date',
        'PERMNO': 'permno'})
            .assign(
        date=lambda x: fix_date(x['date']) + offset)
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .drop_duplicates(subset=['date', 'permno'], keep='last')
            .set_index(['date', 'permno']))


def clean_security(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    columns = [
        'LPERMNO',
        'cusip',
        'datadate',
        'prccm',
        'trt1m',
        'cshoq'
    ]
    return (df[columns]
            .rename(columns={
        'LPERMNO': 'permno',
        'cusip': 'asset_id',
        'prccm': 'prc',
        'cshoq': 'shrout',
        'datadate': 'date'})
            .assign(
        shrout=lambda x: x.groupby('asset_id')['shrout'].ffill() * 1000000,
        asset_id=lambda x: x['asset_id'].apply(lambda s: s[:-1]),
        date=lambda x: fix_date(x['date']))
            .dropna(how='any', subset=['asset_id', 'prc', 'shrout'])
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .drop_duplicates(subset=['date', 'permno'], keep='last')
            .set_index(['date', 'permno']))


def fix_date(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates) + pd.offsets.MonthEnd(0)


# %%
# Stocks Monthly

def merge_assets_factors(df_assets: pd.DataFrame, df_factors: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.merge(left=df_assets,
                         right=df_factors,
                         how='inner',
                         left_index=True,
                         right_index=True)
    df_merged_indexed = (df_merged
                         .reset_index()
                         .drop(columns=['permno'])
                         .assign(date=lambda x: x['date'] - pd.offsets.BQuarterEnd())
                         .drop_duplicates(subset=['date', 'asset_id'], keep='last')
                         .set_index(['date', 'asset_id']))
    log_asset_merge(df_merged_indexed)
    return df_merged_indexed


# Manager / Holdings

def match_fund_manager(df_fund: pd.DataFrame, df_manager: pd.DataFrame, df_key: pd.DataFrame) -> pd.DataFrame:
    df_fund_manager = (df_manager
                       .reset_index()
                       .assign(date=lambda x: x['date'] - pd.offsets.BQuarterEnd())
                       .drop_duplicates(subset=['inv_id', 'date', 'asset_id'], keep='last')
                       .set_index(['inv_id', 'date', 'asset_id']))
    log_holding_merge(df_fund_manager)
    return df_fund_manager


def merge_holding_factor(df_holding: pd.DataFrame, df_asset: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.merge(
        left=df_holding,
        right=df_asset,
        how='left',
        left_index=True,
        right_index=True,
        indicator=True
    )

    df_holding_factor = (df_merged
                         .assign(
        shares=lambda x: np.minimum(x['shares'], x['shrout']),
        ccm_holding=lambda x: x['prc'] * x['shares'],
        ccm_me=lambda x: x['prc'] * x['shrout'],
        holding=lambda x: x['ccm_holding'].fillna(x['backup_holding']) / 1000000,
        me=lambda x: x['ccm_me'].fillna(x['backup_me']) / 1000000,
        typecode=lambda x: x['typecode'].fillna(0))
                         .drop(columns=['ccm_holding', 'backup_holding', 'ccm_me', 'backup_me'])
                         .dropna(subset=['holding', 'me'])
                         .reorder_levels(['inv_id', 'date', 'asset_id']))
    log_holding_factor_merge(df_holding_factor)
    return df_holding_factor


def create_household_sector(df_holding_factor: pd.DataFrame) -> pd.DataFrame:
    df_household = (df_holding_factor
                    .groupby(['date', 'asset_id'])
                    .agg({
        'shares': 'sum',
        'prc': 'last',
        'shrout': 'last',
        'b_mkt': 'last',
        'b_smb': 'last',
        'b_hml': 'last',
        'holding': 'sum',
        'me': 'last',
        '_merge': 'last'})
                    .assign(
        shares=lambda x: np.maximum(x['shrout'] - x['shares'], 0),
        holding=lambda x: np.maximum(x['me'] - x['holding'], 0),
        inv_id=0,
        typecode=0)
                    .set_index('inv_id', append=True)
                    .reorder_levels(['inv_id', 'date', 'asset_id']))
    log_household_sector(df_household)
    df_concat = pd.concat([df_holding_factor, df_household])
    return df_concat


def create_outside_asset(df_household: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    mask = df_household['_merge'] == 'both'
    df_inside = df_household[mask].drop(columns='_merge')
    df_outside = (df_household[~mask]
                  .groupby(['inv_id', 'date'])
                  .agg({
        'typecode': 'last',
        'holding': 'sum'})
                  .assign(asset_id='-1')
                  .set_index('asset_id', append=True))
    log_outside_asset(df_outside)
    return df_inside, df_outside


def construct_zero_holdings(df_inside: pd.DataFrame, n_quarters: int) -> pd.DataFrame:
    def calc_inv_obs(df: pd.DataFrame) -> pd.DataFrame:
        date_diff = (df.groupby('inv_id')['date'].transform('max') - df['date']) / np.timedelta64(1, 'm')
        clean_date_diff = date_diff.astype(int)
        min_diff = np.minimum(clean_date_diff, n_quarters) + 1
        return min_diff

    def calc_asset_obs(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['inv_id', 'asset_id', 'date'])
        next_date = df.groupby(['inv_id', 'asset_id'])['date'].shift(-1)
        date_diff = (next_date - df['date']) / np.timedelta64(1, 'm')
        clean_date_diff = date_diff.fillna(1).astype(int)
        min_diff = np.minimum(clean_date_diff, df['inv_obs'])
        return min_diff

    def date_offset(df: pd.Series) -> pd.Series:
        return df.apply(lambda x: pd.offsets.MonthEnd(x))

    df_holding = (df_inside
                  .reset_index()
                  .assign(
        inv_obs=lambda x: calc_inv_obs(x),
        asset_obs=lambda x: calc_asset_obs(x))
                  .loc[lambda x: x.index.repeat(x['asset_obs'])]
                  .assign(
        n=lambda x: x.groupby(['inv_id', 'date', 'asset_id']).transform('cumcount'),
        mask=lambda x: x['n'] <= 1,
        shares=lambda x: x['shares'] * x['mask'],
        holding=lambda x: x['holding'] * x['mask'],
        date=lambda x: x['date'] + date_offset(x['n']))
                  # .drop(columns=['inv_obs', 'asset_obs', 'mask'])
                  .set_index(['inv_id', 'date', 'n', 'asset_id']))

    log_zero_holdings(df_holding)
    return df_holding


# %%
# Data

def calc_inv_aum(df_inside: pd.DataFrame, df_outside: pd.DataFrame) -> pd.DataFrame:
    df_in_aum = (df_inside
    .groupby(['inv_id', 'date'])
    .agg({
        'holding': 'sum',
        'shares': 'count',
        'typecode': 'last'})
    .rename(columns={
        'holding': 'in_aum',
        'shares': 'n_holding'}))

    df_out_aum = (df_outside
                  .groupby(['inv_id', 'date'])
                  .agg({
        'holding': 'sum',
        'typecode': 'last'})
                  .rename(columns={'holding': 'out_aum'}))

    df_merged = pd.merge(
        left=df_in_aum,
        right=df_out_aum,
        how='outer',
        left_index=True,
        right_index=True)

    df_inv_aum = (df_merged
                  .reset_index()
                  .assign(
        typecode=lambda x: x['typecode_x'].fillna(x['typecode_y']),
        n_holding=lambda x: x['n_holding'].fillna(0),
        out_aum=lambda x: x['out_aum'].fillna(0),
        in_aum=lambda x: x['in_aum'].fillna(0),
        aum=lambda x: x['out_aum'] + x['in_aum'],
        out_weight=lambda x: x['out_aum'] / x['aum'],
        hh_mask=lambda x: (x['inv_id'] == 0) | (x['out_aum'] == 0) | (x['in_aum'] == 0) | (x['aum'] < 10),
        inv_id=lambda x: x['hh_mask'] * x['inv_id'],
        equal_alloc=lambda x: x['hh_mask'] * x['aum'] / (1 + x['n_holding']))
                  .drop(columns=['typecode_x', 'typecode_y', 'hh_mask'])
                  .set_index(['inv_id', 'date']))

    log_inv_aum(df_inv_aum)
    return df_inv_aum


def bin_concentrated_inv(df_inside: pd.DataFrame, df_inv_aum: pd.DataFrame, min_n_holding: int) -> (
pd.DataFrame, pd.DataFrame):
    log_bins(df_inside, df_inv_aum)
    return df_inside, df_inv_aum
    household_mask = (df_inv_aum['out_aum'] == 0) | (df_inv_aum['in_aum'] == 0) | (df_inv_aum['aum'] < 10)
    df_valid = (df_inv_aum
                .reset_index()
                .assign(
        inv_id=lambda x: x['inv_id'].mask(household_mask.values, '0'),
        typecode=lambda x: x['typecode'].mask(household_mask.values, 0))
                .set_index(['inv_id', 'date']))

    diversified_mask = df_valid['n_holding'] >= min_n_holding
    df_aum_diversified = df_valid[diversified_mask]

    def calc_bin(df_date_type: pd.DataFrame) -> pd.Series:
        typecode = df_date_type['typecode'].iloc[0]
        n_bins = np.ceil(df_date_type['n_holding'].sum() / (2 * min_n_holding)).astype(int)
        if (n_bins <= 1) or (typecode == 0):
            return pd.Series('0', index=df_date_type.index)
        else:
            return pd.qcut(x=df_date_type['aum'], q=n_bins, labels=False).apply(fix_qcut_bin, args=(str(typecode),))

    def fix_qcut_bin(bin: int, typecode: str) -> str:
        return typecode + ':' + str(bin)

    df_aum_concentrated = df_valid[~diversified_mask].assign(
        bin=lambda x: x.groupby(['date', 'typecode']).apply(calc_bin).reset_index(drop=True))
    df_concentrated_binned = (df_aum_concentrated
                              .groupby(['bin', 'date'])
                              .agg({
        'in_aum': 'sum',
        'out_aum': 'sum',
        'aum': 'sum',
        'typecode': 'last'})
                              .rename_axis(index={'bin': 'inv_id'}))
    df_aum_binned = (pd.concat([
        df_aum_diversified,
        df_concentrated_binned])
    .assign(
        out_weight=lambda x: x['out_aum'] / x['aum']))
    df_inside_merged = pd.merge(
        left=df_inside,
        right=df_aum_concentrated['bin'],
        how='left',
        left_index=True,
        right_index=True)
    df_inside_binned = (df_inside_merged
                        .reset_index()
                        .assign(bin=lambda x: x['bin'].fillna(x['inv_id']))
                        .groupby(['bin', 'date', 'asset_id'])
                        .agg({
        'holding': 'sum',
        'prc': 'last',
        'shrout': 'last',
        'me': 'last',
        'b_mkt': 'last',
        'b_smb': 'last',
        'b_hml': 'last'})
                        .rename_axis(index={'bin': 'inv_id'}))
    log_bins(df_inside_binned, df_aum_binned)
    return df_inside_binned, df_aum_binned


def assetid_byinv(df_holding: pd.DataFrame) -> pd.Series:
    return df_holding.reset_index('asset_id')['asset_id']


def calc_inv_universe(df_holding: pd.DataFrame, n_quarters: int) -> pd.DataFrame:
    df_assetid_byinv = assetid_byinv(df_holding).sort_index()
    idx_inv_universe = df_assetid_byinv.index.unique()
    df_inv_universe = pd.DataFrame(index=idx_inv_universe, columns=['inv_universe'])
    offset = pd.DateOffset(months=3 * n_quarters)

    def calc_past_quarters(i: int, d: pd.Timestamp) -> np.array:
        prev_date = d - offset
        asset_id_within_range = df_assetid_byinv.loc[i].loc[prev_date:d]
        inv_uni = asset_id_within_range.unique().tolist()
        return inv_uni

    for (inv_id, date) in idx_inv_universe.to_flat_index():
        inv_uni = calc_past_quarters(inv_id, date)
        df_inv_universe.loc[(inv_id, date), 'inv_universe'] = inv_uni

    df_inv_universe = df_inv_universe.assign(uni_size=lambda x: x['inv_universe'].apply(len))
    log_inv_universe(df_inv_universe)
    return df_inv_universe


def create_equal_allocation(df_inv_universe: pd.DataFrame, df_aum_binned: pd.DataFrame) -> pd.DataFrame:
    df_equal_alloc = pd.merge(
        left=df_inv_universe,
        right=df_aum_binned,
        how='inner',
        left_index=True,
        right_index=True)
    return (df_equal_alloc
            .assign(
        valid=lambda x: x.index.get_level_values('inv_id') != '0',
        allocation=lambda x: x['aum'] / (x['uni_size'] + 1))
            .explode('inv_universe')
            .rename(columns={'inv_universe': 'asset_id'})
            .set_index('asset_id', append=True))


def create_total_allocation(df: pd.DataFrame) -> pd.DataFrame:
    return (df['allocation']
            .groupby('asset_id')
            .sum())


def create_instrument(df_inv_universe: pd.DataFrame, df_aum_binned: pd.DataFrame) -> pd.DataFrame:
    df_equal_allocation = create_equal_allocation(df_inv_universe, df_aum_binned)
    total_allocation = create_total_allocation(df_equal_allocation)
    df_instrument = (df_equal_allocation
                     .assign(iv_me=lambda x: total_allocation - x['allocation'])
                     .drop(columns=['allocation', 'valid', 'typecode']))
    log_instrument(df_instrument)
    return df_instrument


def calc_instrument(df_inside_binned: pd.DataFrame, df_aum_binned: pd.DataFrame) -> pd.DataFrame:
    df_merged = pd.merge(
        left=df_inside_binned,
        right=df_aum_binned,
        how='inner',
        left_index=True,
        right_index=True)

    df_instrument = (df_merged
                     .assign(
        total_alloc=lambda x: x.groupby(['date', 'asset_id']).agg({'equal_alloc': 'sum'}),
        iv_me=lambda x: x['equal_alloc'] - x['total_alloc'])
                     .drop(columns=['total_alloc']))
    print(df_instrument.describe())
    log_instrument(df_instrument)
    return df_instrument


# %%
# Estimation

def calc_holding_weights(df_instrument: pd.DataFrame, df_inside_binned: pd.DataFrame, min_holding: int) -> pd.DataFrame:
    mask = (df_instrument['n_holding'] >= min_holding) & (df_instrument['iv_me'] > 0)

    df_weights = (df_instrument.loc[mask]
                  .assign(
        ln_me=lambda x: np.log(x['me']),
        ln_iv_me=lambda x: np.log(x['iv_me']),
        weight=lambda x: x['holding'] / x['aum'],
        rweight=lambda x: x['weight'] / x['out_weight'],
        ln_rweight=lambda x: np.log(x['rweight']),
        mean_ln_rweight=lambda x: x['ln_rweight'].groupby(['inv_id', 'date']).transform('mean'),
        const=1)
                  .drop(columns=['me', 'iv_me', 'weight', 'rweight'])
                  .reset_index('asset_id'))

    log_holding_weights(df_weights)
    return df_weights


def unpack_result(result: gmm.GMMResults) -> dict:
    return result.params


def momcond_1(params, exog):
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


def momcond_2(params, exog):
    upper_bound = 0.9999
    exog = exog.T

    beta_ln_me = params[0]
    beta_characteristics = params[1:]

    ln_me = exog[0]
    rweight = exog[1]
    mean_ln_rweight = exog[2]
    arr_characteristics = exog[3:]

    ln_me_term = (upper_bound - np.exp(-1 * beta_ln_me)) * ln_me
    characteristics_term = np.dot(beta_characteristics, arr_characteristics)
    pred_weight = np.exp(-1 * (ln_me_term + characteristics_term + mean_ln_rweight))

    return rweight * pred_weight


def estimate_model(df_weights: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    def fit_inv_date_1(df_inv_date: pd.DataFrame) -> gmm.GMMResults:
        exog = np.asarray(df_inv_date[['ln_me', 'mean_ln_rweight'] + characteristics])
        instrument = np.asarray(df_inv_date[['ln_iv_me', 'mean_ln_rweight'] + characteristics])
        n = exog.shape[0]
        endog = np.asarray(df_inv_date['ln_rweight'])
        start_params = np.zeros(len(params))
        w0inv = np.dot(instrument.T, instrument) / n

        try:
            model = gmm.NonlinearIVGMM(
                endog=endog,
                exog=exog,
                instrument=instrument,
                func=momcond_1)
            result = model.fit(
                start_params=start_params,
                maxiter=0,
                inv_weights=w0inv)
            # log_results(result, params)
            return result
        except np.linalg.LinAlgError:
            print('Linear Algebra Error')
            print(f'Investor Id:  {df_inv_date.index.get_level_values(0).unique()}')
            print(f'Date:  {df_inv_date.index.get_level_values(1).unique()}')
            return np.NaN

    def fit_inv_date_2(df_inv_date: pd.DataFrame) -> gmm.GMMResults:
        exog = np.asarray(df_inv_date[['ln_me', 'rweight', 'mean_ln_rweight'] + characteristics])
        instrument = np.asarray(df_inv_date[['ln_iv_me', 'rweight', 'mean_ln_rweight'] + characteristics])
        n = exog.shape[0]
        endog = np.ones(n)

        try:
            model = gmm.NonlinearIVGMM(
                endog=endog,
                exog=exog,
                instrument=instrument,
                func=momcond_1)
            w0inv = np.dot(instrument.T, instrument) / n
            start_params = np.zeros(len(params))
            result = model.fit(
                start_params=start_params,
                maxiter=0,
                inv_weights=w0inv)
            log_results(result, params)
            return result
        except np.linalg.LinAlgError:
            print('Singular Matrix Error')
            print(f'Investor Id:  {df_inv_date.index.get_level_values(0).unique()}')
            print(f'Date:  {df_inv_date.index.get_level_values(1).unique()}')
            return np.NaN

    df_model = (df_weights
    .assign(
        gmm_result=lambda x: x.groupby(['inv_id', 'date']).apply(fit_inv_date_2),
        lst_params=lambda x: x['gmm_result'].apply(unpack_result)))
    df_model[params] = pd.DataFrame(df_model['lst_params'].tolist(), index=df_model.index)
    df_model = df_model.drop(columns='lst_params')

    log_params(df_model)
    return df_model


def calc_latent_demand(df_model: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    upper_bound = 0.9999

    df_results = df_model.assign(
        beta_ln_me=lambda x: upper_bound - np.exp(-1 * x['beta_ln_me']),
        beta_const=lambda x: x['beta_const'] + x['mean_ln_rweight'],
        pred_ln_rweight=lambda x: np.einsum('ij,ij->i', x[['ln_me'] + characteristics], x[params]),
        latent_demand=lambda x: x['ln_rweight'] - x['pred_ln_rweight'])
    log_latent_demand(df_results)
    return df_results


# %%
# Liquidity


# %%
# Figures

def check_moment_condition(df_results: pd.DataFrame):
    df_mom = (df_results
              .groupby(['inv_id', 'date'])
              .agg({'latent_demand': 'mean'}))

    epsilon = 0.0001
    mask = (df_mom['latent_demand'] > -1 * epsilon) & (df_mom['latent_demand'] < epsilon)
    valid_rate = len(df_mom[mask]) / len(df_mom)
    print(f'Percentage of valid portfolios:  {100 * valid_rate:.4f}%')

    g = sns.displot(
        data=df_mom,
        x='latent_demand'
    )
    g.set_axis_labels('Log Latent Demand', 'Density')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'moment_condition.png'))


def critical_value_test(df_results: pd.DataFrame, characteristics: list, figure_path: str):
    def iv_reg(df_inv_date: pd.DataFrame):
        y = df_inv_date['ln_me']
        X = df_inv_date[['ln_iv_me'] + characteristics]
        model = OLS(y, X)
        result = model.fit()
        t_stat = result.tvalues.iloc[0]
        return t_stat

    df_iv = (df_results
             .groupby(['inv_id', 'date'])
             .apply(iv_reg)
             .to_frame('t_stat')
             .groupby('date')
             .min()
             .reset_index())

    g = sns.relplot(
        data=df_iv,
        x='date',
        y='t_stat',
        kind='line')
    g.refline(
        y=4.05,
        linestyle='--')
    g.set_axis_labels('Date', 'First stage t-statistic')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'instrument_validity.png'))


def typecode_share_counts(df_inv_aum: pd.DataFrame, figure_path: str):
    df_type_share = (df_inv_aum
                     .assign(typecode=lambda x: x['typecode'].apply(get_readable_typecode))
                     .groupby(['typecode', 'date'])
                     .agg({'aum': 'sum', 'n_holding': 'count'})
                     .assign(share=lambda x: x['aum'] / x['aum'].groupby('date').transform('sum')))

    g = sns.relplot(data=df_type_share, x='date', y='n_holding', hue='typecode', kind='line')
    g.set_axis_labels('Date', 'Number of Investors')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'typecode_count.png'))

    g = sns.relplot(data=df_type_share, x='date', y='share', hue='typecode', kind='line')
    g.set_axis_labels('Date', 'Share of AUM')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'typecode_share.png'))


def test_index_fund(df_results: pd.DataFrame, characteristics: list, params: list, figure_path: str):
    index_id = '90457'
    mask = df_results.index.get_level_values(0) == index_id
    df_index_fund = (df_results
    .assign(
        ln_rweight=lambda x: x['ln_me'] + x['mean_ln_rweight'])
    .loc[mask, ['ln_me', 'ln_iv_me', 'ln_rweight', 'mean_ln_rweight'] + characteristics + params])

    df_index_fund_model = estimate_model(df_index_fund, characteristics, params)
    df_index_fund_result = calc_latent_demand(df_index_fund_model, characteristics, params)
    cols = params + ['latent_demand']
    for param in cols:
        g = sns.relplot(data=df_index_fund_result[[param]],
                        x='date',
                        y=param,
                        kind='line')
        g.set_axis_labels('Date', f'{get_readable_param(param)}')
        g.despine()
        plt.ylim(-1, 1)
        plt.savefig(os.path.join(figure_path, f'index_fund_{param}.png'))


def graph_type_params(df_results: pd.DataFrame, params: list, figure_path: str):
    cols = params + ['latent_demand']

    df_types = df_results.copy()
    df_types[cols] = df_types[cols].apply(lambda x: x * df_results['aum'])
    df_types = (df_types
                .groupby(['typecode', 'date'])
                [cols]
                .sum()
                .apply(lambda x: x / df_results.groupby(['typecode', 'date'])['aum'].sum())
                .reset_index()
                .assign(typecode=lambda x: x['typecode'].apply(get_readable_typecode)))

    for param in cols:
        g = sns.relplot(data=df_types,
                        x='date',
                        y=param,
                        hue='typecode',
                        kind='line')
        g.set_axis_labels('Date', f'{get_readable_param(param)}')
        g.legend.set_title('Institution Type')
        g.despine()
        plt.savefig(os.path.join(figure_path, f'{param}.png'))


def graph_std_latent_demand(df_results: pd.DataFrame, figure_path: str):
    df_ld = (df_results
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
    .assign(
        latent_demand=lambda x: x['latent_demand'] / x['aum'],
        typecode=lambda x: x['typecode'].apply(get_readable_typecode)))

    g = sns.relplot(data=df_ld,
                    x='date',
                    y='latent_demand',
                    hue='typecode',
                    kind='line')
    g.set_axis_labels('Date', 'Standard Deviation of Latent Demand')
    g.legend.set_title('Institution Type')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'std_latent_demand.png'))


def create_tables(df_instrument: pd.DataFrame, df_inv_uni: pd.DataFrame):
    df_pctile = df_inv_uni.assign(pctile=0)
    arr_dates = df_pctile.index.get_level_values('date').unique()

    for date in arr_dates:
        data = df_pctile.loc[date, 'aum']
        df_pctile.loc[date, 'pctile'] = pd.qcut(data, q=100)

    df_grouped_pctile = (df_pctile['uni_persistence']
                         .groupby(['pctile'])
                         .median())
    print(df_grouped_pctile.head(10))


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
    dateindex_s12 = df_s12_clean.index.get_level_values('date')
    print('Cleaned s12')
    print('Number of holdings:  ', len(df_s12_clean))
    print('Earliest date:  ', min(dateindex_s12))
    print('Latest date:  ', max(dateindex_s12))


def log_clean_s12type5(df_s12type5_clean: pd.DataFrame):
    dateindex_s12type5 = df_s12type5_clean.index.get_level_values('date')
    print()
    print('Cleaned s12type5')
    print('Number of firm/dates:  ', len(df_s12type5_clean))
    print('Earliest date:  ', min(dateindex_s12type5))
    print('Latest date:  ', max(dateindex_s12type5))


def log_clean_s34(df_s34_clean: pd.DataFrame):
    dateindex_s34 = df_s34_clean.index.get_level_values('date')
    print()
    print('Cleaned s34')
    print('Number of holdings:  ', len(df_s34_clean))
    print('Earliest date:  ', min(dateindex_s34))
    print('Latest date:  ', max(dateindex_s34))


def log_clean_beta(df_beta_clean: pd.DataFrame):
    dateindex_beta = df_beta_clean.index.get_level_values('date')
    print()
    print('Cleaned beta')
    print('Number of dates:  ', len(df_beta_clean))
    print('Earliest date:  ', min(dateindex_beta))
    print('Latest date:  ', max(dateindex_beta))


def log_clean_security(df_security_clean: pd.DataFrame):
    dateindex_security = df_security_clean.index.get_level_values('date')
    print()
    print('Cleaned security')
    print('Number of asset/dates:  ', len(df_security_clean))
    print('Earliest date:  ', min(dateindex_security))
    print('Latest date:  ', max(dateindex_security))


def log_holding_merge(df_merged: pd.DataFrame):
    print('Merged s12 and s34')
    print('Number of holdings:  ', len(df_merged))
    print()


def log_asset_merge(df_merged: pd.DataFrame):
    print('Merged assets and factors')
    print('Number of assets/dates:  ', len(df_merged))
    print()


def log_holding_factor_merge(df_merged: pd.DataFrame):
    print('Merged holdings and factors')
    print('Number of assets/dates:  ', len(df_merged))
    print()


def log_household_sector(df_household: pd.DataFrame):
    print('Created household sector')
    print('Number of holdings:  ', len(df_household))


def log_outside_asset(df_outside: pd.DataFrame):
    print('Created outside asset')
    print('Number of holdings:  ', len(df_outside))


def log_zero_holdings(df_holding: pd.DataFrame):
    print('Constructed zero holdings')
    print('Number of non-zero holdings:  ', sum(df_holding['shares'] > 0))
    print('Number of zero holdings:  ', sum(df_holding['shares'] == 0))


def log_inv_aum(df_inv_aum: pd.DataFrame):
    idx = df_inv_aum.index.get_level_values('inv_id').unique()
    print('Calculated investor AUM')
    print('Number of investors:  ', len(idx))


def log_bins(df_inside_binned: pd.DataFrame, df_aum_binned: pd.DataFrame):
    print('Binned investors')
    print('Number of investors by holding:  ', len(df_inside_binned.index.unique('inv_id')))
    print('Number of investors by aum:  ', len(df_aum_binned.index.unique('inv_id')))


def log_inv_universe(df_inv_uni: pd.DataFrame):
    idx = df_inv_uni.index.get_level_values('inv_id').unique()
    print('Created investment universe')
    print('Number of investors:  ', len(idx))
    print('Average investment universe size:  ', len(df_inv_uni['uni_size'].mean()))


def log_instrument(df_instrument: pd.DataFrame):
    print('Created market equity instrument')
    print('Number of valid instruments:  ', sum(df_instrument['iv_me'] > 0))
    print('Number of invalid instruments:  ', sum(df_instrument['iv_me'] == 0))


def log_holding_weights(df_model: pd.DataFrame):
    print('Calculated holding weights')
    print('Number of non-zero holdings:  ', sum(df_model['holding'] > 0))
    print('Number of zero holdings:  ', sum(df_model['holding'] == 0))


def log_results(result, params):
    print(result.summary(yname='Latent demand', xname=params))
    print()


def log_params(df_params: pd.DataFrame):
    print('Estimated parameters')
    print('Number of converged estimations:  ', sum(df_params['gmm_result'].notna()))


def log_latent_demand(df_results: pd.DataFrame):
    print('Calculated latent demand')
    print('Number of investors:  ', len(df_results.index.get_level_values('inv_id').unique()))


# %%
# Global parameters

sns.set_theme(style='ticks', palette=sns.color_palette('hls', 6))

input_path = 'data/'
output_path = 'output/'
figure_path = 'figures/'

os.makedirs(output_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

start_date = pd.Timestamp('2012-01-01')
end_date = pd.Timestamp('2017-12-31')

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
# %%
# Main

print('\n---------------Starting Imports---------------------------\n')
dfs = load_wrds(input_path, start_date, end_date)
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
print('\n---------------Merging Holdings/Factors---------------------------\n')
df_holding_factor = merge_holding_factor(df_fund_manager, df_asset)
# df_holding_factor.to_csv(os.path.join(output_path, 'df_holding_factor.csv'))

print('\n---------------Creating Household Sector---------------------------\n')
df_household = create_household_sector(df_holding_factor)
# df_household.to_csv(os.path.join(output_path, 'df_household.csv'))

print('\n---------------Partitioning Outside Asset---------------------------\n')
df_inside, df_outside = create_outside_asset(df_household)
# df_inside.to_csv(os.path.join(output_path, 'df_inside.csv'))
# df_outside.to_csv(os.path.join(output_path, 'df_outside.csv'))
# %%
print('\n---------------Constructing Zero Holdings---------------------------\n')
df_holding = construct_zero_holdings(df_inside, n_quarters)
# df_holding.to_csv(os.path.join(output_path, 'df_holding.csv'))
# %%
print('\n---------------Calculating Investor AUM---------------------------\n')
df_inv_aum = calc_inv_aum(df_inside, df_outside)
# df_inv_aum.to_csv(os.path.join(output_path, 'df_inv_aum.csv'))

print('\n---------------Pooling Investors By Type/Size---------------------------\n')
df_holding_binned, df_aum_binned = bin_concentrated_inv(df_holding, df_inv_aum, min_n_holding)
# df_holding_binned.to_csv(os.path.join(output_path, 'df_holding_binned.csv'))
# df_aum_binned.to_csv(os.path.join(output_path, 'df_aum_binned.csv'))
# %%
# print('\n---------------Tracking Investment Universe---------------------------\n')
# df_inv_universe = calc_inv_universe(df_holding_binned, n_quarters)
# # df_inv_universe.to_csv(os.path.join(output_path, 'df_inv_universe.csv'))

print('\n---------------Calculating Instrument---------------------------\n')
df_instrument = calc_instrument(df_holding_binned, df_aum_binned)
# df_instrument.to_csv(os.path.join(output_path, 'df_instrument.csv'))
# %%
print('\n---------------Calculating Holding Weights---------------------------\n')
df_weights = calc_holding_weights(df_instrument, min_n_holding)
# df_weights.to_csv(os.path.join(output_path, 'df_weights.csv'))
# %%
print('\n---------------Estimating Demand System---------------------------\n')
df_model = estimate_model(df_weights, characteristics, params)
# df_model.to_csv(os.path.join(output_path, 'df_model.csv'))
# %%
print('\n---------------Calculating Latent Demand---------------------------\n')
df_results = calc_latent_demand(df_model, characteristics, params)
# df_results.to_csv(os.path.join(output_path, 'df_results.csv'))
# %%
print('\n---------------Testing Moment Condition---------------------------\n')
check_moment_condition(df_results)
# %%
print('\n---------------Investors by Typecode---------------------------\n')
typecode_share_counts(df_inv_aum, figure_path)
# %%
print('\n---------------Testing Instrument Validity---------------------------\n')
critical_value_test(df_results, characteristics, figure_path)
# %%
print('\n---------------Testing Hypothetical Index Fund---------------------------\n')
test_index_fund(df_results, characteristics, params, figure_path)
# %%
print('\n---------------Parameters By Typecode---------------------------\n')
graph_type_params(df_results, params, figure_path)
# %%
print('\n---------------Standard Deviation of Latent Demand By Typecode---------------------------\n')
graph_std_latent_demand(df_results, figure_path)
# %%

# %%
print('\n---------------Finished---------------------------\n')