# %%
import pandas as pd
import numpy as np
import numpy_financial as npf
from matplotlib import pyplot as plt
import seaborn as sns
import dask.dataframe as dd
from dask.distributed import Client
import os
from statsmodels.sandbox.regression import gmm
from statsmodels.api import OLS


# %%
# Import data

def read_emaxx(input_path: str, start_date: pd.Period, end_date: pd.Period):
    period_range = pd.period_range(start_date, end_date, freq='Q')
    prefix = 'eMAXX\\ASCII_NA_Mkt_All_Pipe_RY_'

    holding_cols = [
        'CUSIP',
        'CUSIPSUFF',
        'FUNDID',
        'BOOL',
        'PARAMT',
    ]
    df_emaxx_holding = []

    fund_cols = [
        'FUNDID',
        'FUNDCLASS',
    ]
    df_emaxx_fund = []

    secmast_cols = [
        'ISSUERCUS',
        'ISSUESUF',
        'CPNRATE',
        'MATDATE',
    ]
    df_emaxx_secmast = []

    issuer_cols = [
        'ISSUERCUS',
        'GEOCODE',
        'ENTITY',
        'STATE'
    ]
    df_emaxx_issuer = []

    broktran_cols = [
        'CUSIP',
        'TRANDATE',
        'ACTCOST',
        'PARAMT',
    ]
    df_emaxx_broktran = []

    for q in period_range:
        log_quarter(q)
        path = os.path.join(input_path, prefix + str(q))

        holding_path = os.path.join(path, 'HOLDING.txt')
        holding_data = (pd.read_csv(filepath_or_buffer=holding_path,
                                    delimiter='|',
                                    usecols=holding_cols,
                                    low_memory=False)
                        .assign(date=q))
        df_emaxx_holding.append(holding_data)

        fund_path = os.path.join(path, 'FUND.txt')
        fund_data = (pd.read_csv(filepath_or_buffer=fund_path,
                                 delimiter='|',
                                 usecols=fund_cols,
                                 low_memory=False)
                     .assign(date=q))
        df_emaxx_fund.append(fund_data)

        # secmast_path = os.path.join(path, 'SECMAST.txt')
        # secmast_data = (pd.read_csv(filepath_or_buffer=secmast_path,
        #                             delimiter='|',
        #                             usecols=secmast_cols,
        #                             low_memory=False)
        #                 .assign(date=q))
        # df_emaxx_secmast.append(secmast_data)

        # issuer_path = os.path.join(path, 'ISSUERS.txt')
        # issuer_data = (pd.read_csv(filepath_or_buffer=issuer_path,
        #                          delimiter='|',
        #                          usecols=issuer_cols,
        #                          low_memory=False)
        #              .assign(date=q))
        # df_emaxx_issuer.append(issuer_data)

        # broktran_path = os.path.join(path, 'BROKTRAN.txt')
        # broktran_data = (pd.read_csv(filepath_or_buffer=broktran_path,
        #                              delimiter='|',
        #                              usecols=broktran_cols,
        #                              low_memory=False)
        #                  .assign(date=q)
        #                  .sort_values('TRANDATE')
        #                  .drop_duplicates('CUSIP', keep='last')
        #                  .drop(columns='TRANDATE'))
        # df_emaxx_broktran.append(broktran_data)

    df_emaxx_holding = pd.concat(df_emaxx_holding)
    df_emaxx_fund = pd.concat(df_emaxx_fund)
    # df_emaxx_secmast = pd.concat(df_emaxx_secmast)
    # df_emaxx_issuer = pd.concat(df_emaxx_issuer)
    # df_emaxx_broktran = pd.concat(df_emaxx_broktran)

    return df_emaxx_holding, df_emaxx_fund, df_emaxx_secmast, df_emaxx_issuer, df_emaxx_broktran


def pandas_read_bond(path: str, start_date: pd.Period, end_date: pd.Period):
    df_emaxx_holding, df_emaxx_fund, df_emaxx_secmast, df_emaxx_issuer, df_emaxx_broktran = read_emaxx(path, start_date,
                                                                                                       end_date)
    log_import_emaxx_holding(df_emaxx_holding)
    log_import_emaxx_fund(df_emaxx_fund)
    # log_import_emaxx_secmast(df_emaxx_secmast)
    # log_import_emaxx_issuer(df_emaxx_issuer)
    log_import_emaxx_broktran(df_emaxx_broktran)

    fed_cols = ['cusip',
                'parValue',
                'date']
    df_fed = pd.read_csv(filepath_or_buffer=os.path.join(path, 'FED', 'fed_treasury_holding.csv'),
                         usecols=fed_cols)
    log_import_fed(df_fed)

    df_treasury = pd.read_csv(filepath_or_buffer=os.path.join(path, 'Auctions_TreasurySecurities.csv'))
    log_import_treasury(df_treasury)

    crsp_cols = ['TCUSIP',
                 'TMATDT',
                 'TCOUPRT',
                 'ITYPE',
                 'MCALDT',
                 'TMYLD',
                 'TMBID',
                 'TMASK',
                 'TMNOMPRC',
                 'TMTOTOUT']
    df_crsp = pd.read_csv(filepath_or_buffer=os.path.join(path, 'crsp_treasury.csv'),
                          usecols=crsp_cols,
                          low_memory=False)
    log_import_crsp(df_crsp)

    wrds_cols = ['DATE',
                 'CUSIP',
                 'BOND_TYPE',
                 'AMOUNT_OUTSTANDING',
                 'MATURITY',
                 'COUPON',
                 'T_Yld_Pt',
                 'PRICE_EOM',
                 'R_SP',
                 'N_SP',
                 'T_Spread']
    df_wrds_bond = pd.read_csv(filepath_or_buffer=os.path.join(path, 'wrds_bond.csv'),
                               usecols=wrds_cols,
                               low_memory=False)
    log_import_wrds_bond(df_wrds_bond)

    return df_emaxx_holding, df_emaxx_fund, df_emaxx_secmast, df_emaxx_issuer, df_emaxx_broktran, df_fed, df_treasury, df_crsp, df_wrds_bond


def clean_imports_bond(df_emaxx_holding: pd.DataFrame,
                       df_emaxx_fund: pd.DataFrame,
                       df_emaxx_secmast: pd.DataFrame,
                       df_emaxx_issuer: pd.DataFrame,
                       df_emaxx_broktran: pd.DataFrame,
                       df_fed: pd.DataFrame,
                       df_treasury: pd.DataFrame,
                       df_crsp: pd.DataFrame,
                       df_wrds_bond: pd.DataFrame,
                       start_date: pd.Period,
                       end_date: pd.Period) -> tuple:
    df_emaxx_holding_clean = clean_emaxx_holding(df_emaxx_holding)
    log_clean_emaxx_holding(df_emaxx_holding_clean)

    df_emaxx_fund_clean = clean_emaxx_fund(df_emaxx_fund)
    log_clean_emaxx_fund(df_emaxx_fund_clean)

    # df_emaxx_secmast_clean = clean_emaxx_secmast(df_emaxx_secmast)
    # log_clean_emaxx_secmast(df_emaxx_secmast_clean)

    # df_emaxx_issuer_clean = clean_emaxx_issuer(df_emaxx_issuer)
    # log_clean_emaxx_issuer(df_emaxx_issuer_clean)

    # df_emaxx_broktran_clean = clean_emaxx_broktran(df_emaxx_broktran)
    # log_clean_emaxx_broktran(df_emaxx_broktran_clean)

    df_fed_clean = clean_fed_holding(df_fed, start_date, end_date)
    log_clean_fed(df_fed_clean)

    df_treasury_clean = clean_treasury(df_treasury)
    log_clean_treasury(df_treasury_clean)

    df_crsp_clean = clean_crsp(df_crsp, start_date, end_date)
    log_clean_crsp(df_crsp_clean)

    df_wrds_bond_clean = clean_wrds_bond(df_wrds_bond, start_date, end_date)
    log_clean_wrds_bond(df_wrds_bond_clean)

    return df_emaxx_holding_clean, df_emaxx_fund_clean, df_fed_clean, df_treasury_clean, df_crsp_clean, df_wrds_bond_clean


def clean_emaxx_holding(df_emaxx_holding: pd.DataFrame) -> pd.DataFrame:
    df_emaxx_holding.columns = df_emaxx_holding.columns.str.lower()
    return (df_emaxx_holding
            .rename(columns={'cusip': 'issuer_id',
                             'cusipsuff': 'issue',
                             'fundid': 'inv_id',
                             'bool': 'q_date'})
            .assign(asset_id=lambda x: x['issuer_id'] + x['issue'])
            .dropna(subset=['asset_id'])
            .drop(columns=['issue', 'issuer_id'])
            .sort_values('paramt')
            .drop_duplicates(subset=['inv_id', 'date', 'asset_id'], keep='last')
            .reset_index(drop=True))


def clean_emaxx_fund(df_emaxx_fund: pd.DataFrame) -> pd.DataFrame:
    df_emaxx_fund.columns = df_emaxx_fund.columns.str.lower()
    return (df_emaxx_fund
            .rename(columns={'fundid': 'inv_id',
                             'fundclass': 'typecode'})
            .dropna(subset=['typecode', 'inv_id'])
            .assign(typecode=lambda x: x['typecode'].apply(agg_typcode))
            .drop_duplicates(subset=['inv_id', 'date'], keep='last')
            .reset_index(drop=True))


def clean_emaxx_secmast(df_emaxx_secmast: pd.DataFrame) -> pd.DataFrame:
    df_emaxx_secmast.columns = df_emaxx_secmast.columns.str.lower()
    return (df_emaxx_secmast
            .rename(columns={'cpnrate': 'coupon'})
            .loc[lambda x: x['matdate'] != 99990101]
            .dropna(subset=['issuercus', 'issuesuf'])
            .assign(asset_id=lambda x: x['issuercus'] + x['issuesuf'],
                    matdate=lambda x: pd.to_datetime(x['matdate'], format='%Y%m%d').dt.to_period('Q'),
                    maturity=lambda x: (x['date'] - x['matdate']).apply(lambda y: y.n))
            .drop(columns=['cusip', 'cusipsuff'])
            .reset_index(drop=True))


def clean_emaxx_issuer(df_emaxx_issuer: pd.DataFrame) -> pd.DataFrame:
    df_emaxx_issuer.columns = df_emaxx_issuer.columns.str.lower()
    return (df_emaxx_issuer
            .dropna(subset=['issuercus', 'geocode', 'entity'])
            .rename(columns={'issuercus': 'issuer_id'})
            .astype({'issuer_id': str})
            .reset_index(drop=True))


def clean_emaxx_broktran(df_emaxx_broktran: pd.DataFrame) -> pd.DataFrame:
    df_emaxx_broktran.columns = df_emaxx_broktran.columns.str.lower()
    return (df_emaxx_broktran
            .rename(columns={'cusip': 'asset_id'})
            .assign(prc=lambda x: x['actcost'] / x['paramt'])
            .drop(columns=['actcost', 'paramt'])
            .reset_index(drop=True))


def clean_fed_holding(df_fed_holding: pd.DataFrame, start_date: pd.Period, end_date: pd.Period) -> pd.DataFrame:
    return (df_fed_holding
    .rename(columns={'cusip': 'asset_id',
                     'maturityDate': 'maturity',
                     'parValue': 'paramt'})
    .assign(asset_id=lambda x: x['asset_id'].apply(lambda s: s[:-1]),
            paramt=lambda x: x['paramt'] / 1000000,
            inv_id=-1,
            typecode='FED',
            date=lambda x: x['date'].apply(lambda y: pd.Period(y, freq='Q')))
    .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)])


def clean_treasury(df_treasury: pd.DataFrame) -> pd.DataFrame:
    df_treasury.columns = df_treasury.columns.str.lower()
    df_treasury.columns = df_treasury.columns.str.replace(' ', '_')
    return (df_treasury
            .rename(columns={'cusip': 'asset_id',
                             'interest_rate': 'coupon'})
            .loc[lambda x: x['floating_rate'] == 'No']
            .assign(coupon=lambda x: x['coupon'].fillna(0),
                    amtout=lambda x: x['currently_outstanding'].fillna(x['total_accepted']))
            .dropna(subset=['amtout', 'coupon'])
            .sort_values('issue_date')
            .drop_duplicates('asset_id', keep='last')
            .drop(columns=['floating_rate'])
            .reset_index(drop=True))


def clean_crsp(df_crsp: pd.DataFrame, start_date: pd.Period, end_date: pd.Period) -> pd.DataFrame:
    df_crsp.columns = df_crsp.columns.str.lower()
    return (df_crsp
            .rename(columns={'tcusip': 'asset_id',
                             'mcaldt': 'date',
                             'tmatdt': 'maturity',
                             'tcouprt': 'coupon',
                             'tmyld': 'ytm',
                             'tmnomprc': 'prc',
                             'tmtotout': 'amtout'})
            .loc[lambda x: (x['itype'] >= 1) & (x['itype'] <= 4)]
            .assign(date=lambda x: fix_date_quarterly(x['date']),
                    ytm=lambda x: x['ytm'] * 100,
                    credit_rating=1)
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .dropna(subset=['asset_id', 'prc', 'amtout'])
            .sort_values('date')
            .drop_duplicates(subset=['date', 'asset_id'], keep='last')
            .drop(columns=['itype'])
            .reset_index(drop=True))


def clean_wrds_bond(df_wrds_bond: pd.DataFrame, start_date: pd.Period, end_date: pd.Period) -> pd.DataFrame:
    df_wrds_bond.columns = df_wrds_bond.columns.str.lower()
    return (df_wrds_bond
            .loc[lambda x: x['bond_type'] != 'CCOV']
            .rename(columns={'cusip': 'asset_id',
                             'price_eom': 'prc',
                             't_yld_pt': 'ytm',
                             't_spread': 'spread',
                             'n_sp': 'credit_rating',
                             'amount_outstanding': 'amtout'})
            .assign(date=lambda x: fix_date_quarterly(x['date']),
                    asset_id=lambda x: x['asset_id'].apply(lambda s: s[:-1]),
                    spread=lambda x: x['spread'].replace('%', '', regex=True).astype(float),
                    is_treasury=0)
            .loc[lambda x: (x['date'] >= start_date) & (x['date'] <= end_date)]
            .dropna(subset=['asset_id', 'prc'])
            .drop(columns=['bond_type'])
            .sort_values('date')
            .drop_duplicates(subset=['date', 'asset_id'])
            .reset_index(drop=True))


def fix_date_quarterly(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period(freq='Q')


def fix_date_monthly(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.to_period(freq='M')


# %%
# Bonds Monthly

def calc_factors(df_crsp_clean: pd.DataFrame, df_treasury_clean: pd.DataFrame, end_date: pd.Period) -> pd.DataFrame:
    df_otr = (df_treasury_clean
              .sort_values('issue_date')
              .assign(otf_date=(df_treasury_clean
                                .groupby(['security_term'])
                                ['issue_date']
                                .shift(periods=-1,
                                       fill_value=end_date + 1))))

    # df_factor = (df_crsp_clean
    #              .merge(right=df_otr,
    #                     how='left',
    #                     on=['asset_id', 'date']))
    df_factor = (df_crsp_clean
                 .assign(spread=lambda x: 100 * (x['tmask'] - x['tmbid']) / x['tmask'],
                         is_treasury=1)
                 .drop(columns=['tmask', 'tmbid']))

    log_factors(df_factor)
    return df_factor


def merge_corp_treasury(df_factor: pd.DataFrame, df_wrds_bond_clean: pd.DataFrame) -> pd.DataFrame:
    df_asset = (pd.concat([df_wrds_bond_clean, df_factor], ignore_index=True)
                .assign(maturity=lambda x: pd.to_datetime(x['maturity'], format='%Y-%m-%d'))
                .drop_duplicates(['asset_id', 'date'], keep='last'))
    log_corp_treasury(df_asset)
    return df_asset


def merge_holding_fund(df_emaxx_holding_clean: pd.DataFrame, df_emaxx_fund_clean: pd.DataFrame) -> pd.DataFrame:
    df_holding_fund = (df_emaxx_holding_clean
                       .merge(right=df_emaxx_fund_clean,
                              how='left',
                              on=['date', 'inv_id']))

    log_holding_fund_merge(df_holding_fund)
    return df_holding_fund


def merge_fed_holding(df_emaxx_holding_clean: pd.DataFrame, df_fed_clean: pd.DataFrame) -> pd.DataFrame:
    df_fed_holding = pd.concat([df_emaxx_holding_clean, df_fed_clean], ignore_index=True)
    log_fed_holding(df_fed_holding)
    return df_fed_holding


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
                          paramt=lambda x: x['paramt'] * x['mask'],
                          date=lambda x: x['date'] + x['asset_obs'])
                  .drop(columns=['inv_obs', 'asset_obs', 'mask']))

    log_zero_holdings(df_holding)
    return df_holding


def merge_holding_factor(df_holding_fed: pd.DataFrame, df_asset: pd.DataFrame) -> pd.DataFrame:
    df_holding_factor = (df_holding_fed
                         .merge(right=df_asset,
                                how='left',
                                on=['date', 'asset_id'])
                         .assign(holding=lambda x: x['prc'] * x['paramt'] / 100000,
                                 q_date=lambda x: pd.to_datetime(x['q_date'], format='%Y%m%d'),
                                 t_maturity=lambda x: (x['maturity'] - x['q_date']).dt.days.fillna(0) / 30.5)
                         .loc[lambda x: x['amtout'] > 0]
                         .dropna(subset=['holding'])
                         .drop(columns=['q_date', 'maturity']))

    log_holding_factor_merge(df_holding_factor)
    return df_holding_factor


def calc_maturity(df_holding_factor: pd.DataFrame) -> pd.DataFrame:
    return (df_holding_factor)


def drop_unused_inv_asset(df_holding_factor: pd.DataFrame) -> pd.DataFrame:
    df_dropped = (df_holding_factor
                  .assign(Tinv_paramt=lambda x: x.groupby(['inv_id', 'date'])['paramt'].transform('sum'),
                          Tasset_paramt=lambda x: x.groupby(['date', 'asset_id'])['paramt'].transform('sum'),
                          mask=lambda x: (x['Tinv_paramt'] == 0) | (x['Tasset_paramt'] == 0))
                  .loc[lambda x: ~x['mask']]
                  .drop(columns=['Tinv_paramt', 'Tasset_paramt', 'mask']))

    log_drop_unused(df_dropped)
    return df_dropped


def create_household_sector(df_dropped: pd.DataFrame) -> pd.DataFrame:
    df_household = (df_dropped
                    .groupby(['date', 'asset_id'], as_index=False)
                    .agg({
        'paramt': 'sum',
        'prc': 'last',
        'amtout': 'last',
        'holding': 'sum',
        'ytm': 'last'})
                    .assign(paramt=lambda x: np.maximum(x['amtout'] - x['paramt'], 0),
                            holding=lambda x: x['holding'] * x['prc'],
                            inv_id=0,
                            typecode='HOU'))

    log_household_sector(df_household)
    df_concat = pd.concat([df_dropped, df_household], ignore_index=True)
    return df_concat


# %%
# Data

def get_treasury_id(df_emaxx_issuer_clean: pd.DataFrame) -> np.array:
    df_t = (df_emaxx_issuer_clean
    .assign(usa_mask=lambda x: x['geocode'] == 'USA',
            gov_mask=lambda x: x['entity'] == 'GT',
            t_mask=lambda x: x['usa_mask'] & x['gov_mask'])
    .loc[lambda x: x['treasury_mask']])
    return df_t['issuer_id'].unique()


def partition_outside_asset(df_household: pd.DataFrame) -> pd.DataFrame:
    df_outside = (df_household
                  .assign(out_mask=lambda x: x.isna().any(axis=1),
                          out_holding=lambda x: x['holding'] * x['out_mask']))

    log_outside_asset(df_outside)
    return df_outside


def calc_inv_aum(df_outside: pd.DataFrame) -> pd.DataFrame:
    df_inv_aum = (df_outside
                  .assign(aum=lambda x: x.groupby(['inv_id', 'date'])['holding'].transform('sum'))
                  .loc[lambda x: x['aum'] > 0]
                  .assign(out_aum=lambda x: x.groupby(['inv_id', 'date'])['out_holding'].transform('sum'))
                  .assign(out_aum=lambda x: x['out_aum'].mask(x['typecode'] == 'FED', 1),
                          out_weight=lambda x: x['out_aum'] / x['aum']))

    log_inv_aum(df_inv_aum)
    return df_inv_aum


def agg_small_inv(df_inv_aum: pd.DataFrame) -> pd.DataFrame:
    df_agg = (df_inv_aum
              .assign(small_mask=lambda x: (x['aum'] < 10) | (x['out_weight'] == 0) | (x['out_weight'] == 1),
                      inv_id=lambda x: ~x['small_mask'] * x['inv_id'],
                      typecode=lambda x: x['typecode'].mask(x['small_mask'], 'HOU'),
                      hh_mask=lambda x: x['inv_id'] == 0))

    df_grouped = df_agg.groupby(['inv_id', 'date', 'asset_id'])

    df_agg = (df_agg
              .assign(Tparamt=lambda x: df_grouped['paramt'].transform('sum'),
                      Tholding=lambda x: df_grouped['holding'].transform('sum'),
                      paramt=lambda x: np.where(x['hh_mask'], x['Tparamt'], x['paramt']),
                      holding=lambda x: np.where(x['hh_mask'], x['Tholding'], x['holding']))
              .drop(columns=['Tparamt', 'Tholding']))

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

        if (n_bins <= 1) | (n_holding >= min_n_holding) | (typecode == 0) | (typecode == 'FED'):
            return pd.Series(0, index=idx)
        else:
            return pd.Series(pd.qcut(x=df_type_date['aum'], q=n_bins, labels=False), index=idx)

    df_binned = (df_inv_aum
                 .reset_index(drop=True)
                 .assign(bin=lambda x: x.groupby(['typecode', 'date']).apply(calc_bin).reset_index(drop=True))
                 .assign(bin=lambda x: x.groupby(['date', 'typecode', 'bin']).ngroup()))

    log_bins(df_binned)
    return df_binned


def calc_ytm(coupon_rate: pd.Series, maturity: pd.Series, prc: pd.Series, par: pd.Series) -> pd.Series:
    coupon = coupon_rate * par / 200
    appreciation = (par - prc) / maturity
    average_prc = (par + prc) / 2
    return (coupon + appreciation) / average_prc


def calc_instrument(df_binned: pd.DataFrame) -> pd.DataFrame:
    df_instrument = (df_binned
                     .assign(total_alloc=lambda x: x.groupby(['date', 'asset_id'])['equal_alloc'].transform('sum'),
                             iv_me=lambda x: x['total_alloc'] - x['equal_alloc'],
                             iv_ytm=lambda x: calc_ytm(x['coupon'], x['t_maturity'], x['iv_me'], x['amtout']))
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
                  .assign(ln_iv_me=lambda x: np.log(x['iv_me']),
                          weight=lambda x: x['holding'] / x['aum'],
                          rweight=lambda x: x['holding'] / x['out_aum'],
                          ln_rweight=lambda x: np.log(x['rweight'].mask(x['rweight'] == 0, np.NaN)),
                          mean_ln_rweight=lambda x: x.groupby(['inv_id', 'date'])['ln_rweight'].transform('mean'),
                          const=1,
                          pct_uni_held=lambda x: np.log(x['n_holding']) - np.log(x['x_holding']))
                  .drop(columns=['iv_me']))

    log_holding_weights(df_weights)
    return df_weights


# %%
# Estimation

def bound_float(arr: np.array) -> float:
    ln_bound = 709.7827
    return np.minimum(np.maximum(arr, -ln_bound), ln_bound)


def momcond_L(params: np.array, exog: np.array) -> np.array:
    lower_bound = -0.999
    exog = exog.T

    beta_yield = params[0]
    beta_characteristics = params[1:]

    ytm = exog[0]
    mean_ln_rweight = exog[1]
    arr_characteristics = exog[2:]

    yield_term = beta_yield * ytm
    characteristics_term = np.dot(beta_characteristics, arr_characteristics)
    pred_weight = yield_term + characteristics_term + mean_ln_rweight

    return pred_weight


def fit_inv_date_L(df_inv_date: pd.DataFrame, characteristics: list, params: list, ) -> gmm.GMMResults:
    df_inv_date = df_inv_date.dropna(subset='ln_rweight')
    exog = np.asarray(df_inv_date[['ytm', 'mean_ln_rweight'] + characteristics])
    instrument = np.asarray(df_inv_date[['iv_ytm', 'mean_ln_rweight'] + characteristics])
    n = exog.shape[0]
    endog = np.asarray(df_inv_date['ln_rweight'] - df_inv_date['pct_uni_held'])
    start_params = np.zeros(len(params))
    w0inv = np.dot(instrument.T, instrument) / n

    if len(exog) == 1:
        print('One valid holding')
        return None

    # try:
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

    # except np.linalg.LinAlgError:
    #     print('Linear Algebra Error')
    #     return None


def estimate_model_L(df_weights: pd.DataFrame, characteristics: list, params: list, min_n_holding: int) -> pd.DataFrame:
    mask = df_weights['n_holding'] >= min_n_holding

    df_institutions = (df_weights
                       .loc[mask]
                       .set_index(['inv_id', 'date'])
                       .assign(gmm_result=lambda x: x.groupby(['inv_id', 'date']).apply(
        lambda y: fit_inv_date_L(y, characteristics, params)).reset_index(drop=True))
                       .reset_index())

    df_bins = (df_weights
               .loc[~mask]
               .set_index(['bin', 'date'])
               .assign(
        gmm_result=lambda x: x.groupby(['bin', 'date']).apply(lambda y: fit_inv_date_L(y, characteristics, params)))
               .reset_index())

    df_model = pd.concat([df_institutions, df_bins], ignore_index=True)
    log_params(df_model)
    return df_model


def momcond_NL(params: np.array, exog: np.array) -> np.array:
    upper_bound = 0.9999
    exog = exog.T

    beta_ytm = bound_float(params[0])
    beta_characteristics = params[1:]

    ytm = exog[0]
    rweight = exog[1]
    mean_ln_rweight = exog[2]
    arr_characteristics = exog[3:]

    ytm_term = beta_ytm * ytm
    characteristics_term = np.dot(beta_characteristics, arr_characteristics)
    pred_ln_rweight = bound_float(ytm_term + characteristics_term + mean_ln_rweight)
    pred_rweight = np.exp(-1 * pred_ln_rweight)

    return rweight * pred_rweight


def fit_inv_date_NL(df_inv_date: pd.DataFrame, characteristics: list, params: list) -> gmm.GMMResults:
    exog = np.asarray(df_inv_date[['ytm', 'rweight', 'mean_ln_rweight'] + characteristics])
    instrument = np.asarray(df_inv_date[['iv_ytm', 'rweight', 'mean_ln_rweight'] + characteristics])
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

    df_model = pd.concat([df_institutions, df_bins], ignore_index=True)
    log_params(df_model)
    return df_model


def calc_latent_demand(df_model: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    def unpack_result(result: gmm.GMMResults) -> list:
        return result.params

    def unpack_params(df: pd.DataFrame) -> pd.DataFrame:
        df[params] = pd.DataFrame(df['lst_params'].tolist(), index=df.index)
        return df

    df_results = (df_model
                  .dropna()
                  .assign(lst_params=lambda x: x['gmm_result'].apply(unpack_result))
                  .pipe(unpack_params)
                  .drop(columns='lst_params')
                  .assign(beta_const=lambda x: x['beta_const'] + x['mean_ln_rweight'],
                          pred_ln_rweight=lambda x: np.einsum('ij,ij->i', x[['ytm'] + characteristics], x[params]),
                          latent_demand=lambda x: x['ln_rweight'] - x['pred_ln_rweight'],
                          moment=lambda x: x['latent_demand'] - x['pct_uni_held'])
                  .drop(columns=['pct_uni_held', 'gmm_result', 'pred_ln_rweight']))

    log_latent_demand(df_results)
    return df_results


# %%
# Liquidity

def calc_coliquidity_matrix(df_date: pd.DataFrame) -> pd.Series:
    df_asset_grouped = df_date.groupby('asset_id')
    n_assets = len(df_asset_grouped)
    print(n_assets)

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
        print('Added asset ', col)

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
                           gap=lambda x: np.log(x['demand']) - x['loop_ln_price'] - x['loop_ln_amtout']))

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
                                     loop_ln_amtout=lambda x: x['lag_ln_amtout'])
        else:
            df_date = df_date.assign(loop_ln_prc=lambda x: x['prc'],
                                     loop_ln_amtout=lambda x: x['amtout'])

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
                loop_latent=lambda x: np.einsum('ij,ij->i', df_date[['loop_ln_amtout'] + loop_chars],
                                                df_date[loop_params]) + x['lag_latent_demand'])
        else:
            df_date = df_date.assign(
                loop_latent=lambda x: np.einsum('ij,ij->i', df_date[['loop_ln_amtout'] + loop_chars],
                                                df_date[loop_params]) + x['latent_demand'])

    return df_date


def decompose_variance(df_results: pd.DataFrame, characteristics: list, params: list) -> pd.DataFrame:
    mask = df_results['holding'] > 0

    df_variance = (df_results
                   .loc[mask]
                   .assign(ln_prc=lambda x: np.log(x['prc']),
                           ln_amtout=lambda x: np.log(x['amtout'])))

    df_grouped = df_variance.groupby(['date', 'asset_id'])

    df_variance = (df_variance
                   .assign(lag_ln_prc=lambda x: df_grouped['ln_prc'].shift(4),
                           lag_ln_amtout=lambda x: df_grouped['ln_amtout'].shift(4),
                           lag_aum=lambda x: df_grouped['aum'].shift(4),
                           lag_latent=lambda x: df_grouped['latent_demand'].shift(4)))

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
            .assign(date=lambda x: x['date'].apply(lambda p: p.strftime(None))))


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

    return df_figure


def check_moment_condition(df_figures: pd.DataFrame, min_n_holding: int):
    mask = df_figures['n_holding'] >= min_n_holding
    df_mom = (df_figures
              .groupby(['inv_id', 'date'])
              .agg({'moment': 'mean'}))

    epsilon = 0.01
    mask = (df_mom['moment'] > -epsilon) & (df_mom['moment'] < epsilon)
    valid_rate = len(df_mom[mask]) / len(df_mom)
    print(f'Percentage of valid moments:  {100 * valid_rate:.4f}%')

    g = sns.displot(
        data=df_mom,
        x='moment'
    )
    g.set_axis_labels('Log Latent Demand', 'Frequency')
    g.despine()
    plt.savefig(os.path.join(figure_path, f'moment_condition.png'))

    return df_figures


def critical_value_test(df_figures: pd.DataFrame, characteristics: list, min_n_holding: int, figure_path: str):
    def iv_reg(df_inv_date: pd.DataFrame):
        y = df_inv_date['ytm']
        X = df_inv_date[['iv_ytm'] + characteristics]
        model = OLS(y, X)
        result = model.fit()
        t_stat = result.tvalues.iloc[0]
        return abs(t_stat)

    df_iv = (df_figures
             .groupby(['inv_id', 'date'])
             .apply(iv_reg)
             .to_frame('t_stat')
             .groupby('date')
             .median()
             .reset_index())

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
    .assign(ln_rweight=lambda x: x['ytm'] + x['mean_ln_rweight'],
            pct_uni_held=1)
    .loc[mask])

    df_index_fund_model = (df_index_fund
                           .set_index(['inv_id', 'date'])
                           .assign(
        gmm_result=lambda x: x.groupby(['inv_id', 'date']).apply(lambda y: fit_inv_date_L(y, characteristics, params)))
                           .reset_index())
    df_index_fund_result = calc_latent_demand(df_index_fund_model, characteristics, params)
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


def graph_type_params(df_figures: pd.DataFrame, params: list, figure_path: str):
    df_types = df_figures.copy()
    df_types[params] = df_types[params].apply(lambda x: x * df_figures['aum'])
    df_types = (df_types
                .groupby(['typecode', 'date'])
                [params]
                .sum()
                .apply(lambda x: x / df_figures.groupby(['typecode', 'date'])['aum'].sum()))

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

    return df_figures


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

    return df_figures


def summary_tables(df_figures: pd.DataFrame):
    return df_figures


def save_df(df: pd.DataFrame, path: str, name: str):
    df.to_csv(os.path.join(path, name), index=False)
    return df


def get_param_cols(cols: list) -> list:
    return ['beta_' + col for col in cols]


def get_readable_param(name: str) -> str:
    return name.replace('_', ' ').title()


def get_readable_typecode(typecode: str):
    dict_typecode = {'TTF': '13F Filer',
                     'FOK': '401K',
                     'ANN': 'Annuity/Variable Annuity',
                     'AMM': 'Annuity/VA - Money Market',
                     'BKP': 'Bank-Portfolio',
                     'SVG': 'Bank-Savings/Bldg Society',
                     'BKT': 'Bank-Trust',
                     'CHU': 'Church/Religious Org',
                     'CRP': 'Corporation',
                     'CRU': 'Credit Union',
                     'FCC': 'Finance Company',
                     'FED': 'Federal Reserve',
                     'FEN': 'Foundation/Endowment',
                     'GVT': 'Government',
                     'HLC': 'Health Care Systems',
                     'HFD': 'Hedge Fund',
                     'HOU': 'Households',
                     'HSP': 'Hospital',
                     'INS': 'Insurance Co-Diversified',
                     'LIN': 'Insurance Co-Life/Health',
                     'PIN': 'Insurance Co-Prop & Cas',
                     'INM': 'Investment Manager',
                     'BAL': 'Mutual Fund - Balanced',
                     'MMM': 'Mutual Fund - Money Mkt',
                     'MUT': 'MutFd-OE/UnitTr/SICAV/FCP',
                     'END': 'MutFd-CE/Inv Tr/FCP',
                     'QUI': 'Mutual Fund-Equity',
                     'FOF': 'Mutual Fund-Fund of Funds',
                     'NDT': 'Nuclear De-Comm Trust',
                     'OTH': 'Other',
                     'CPF': 'Pension Fund-Corporate',
                     'GPE': 'Pension Fund-Government',
                     'UPE': 'Pension Fund-Union',
                     'RIN': 'Reinsurance Company',
                     'SBC': 'Small Business Invst Co',
                     'SPZ': 'Spezial Fund',
                     'UIT': 'Unit Investment Trust'}
    return dict_typecode[typecode]


def agg_typcode(typecode: str):
    dict_type_agg = {'TTF': 'Investment Manager',
                     'FOK': 'Pension Fund',
                     'ANN': 'Variable Annuity',
                     'AMM': 'Variable Annuity',
                     'BKP': 'Bank',
                     'SVG': 'Bank',
                     'BKT': 'Bank',
                     'CHU': 'Other',
                     'CRP': 'Corporation',
                     'CRU': 'Bank',
                     'FCC': 'Investment Manager',
                     'FED': 'Federal Reserve',
                     'FEN': 'Other',
                     'GVT': 'Government',
                     'HLC': 'Insurance Company',
                     'HFD': 'Investment Manager',
                     'HOU': 'Households',
                     'HSP': 'Insurance Company',
                     'INS': 'Insurance Company',
                     'LIN': 'Insurance Company',
                     'PIN': 'Insurance Company',
                     'INM': 'Investment Manager',
                     'BAL': 'Mutual Fund',
                     'MMM': 'Mutual Fund',
                     'MUT': 'Mutual Fund',
                     'END': 'Mutual Fund',
                     'QUI': 'Mutual Fund',
                     'FOF': 'Mutual Fund',
                     'NDT': 'Other',
                     'OTH': 'Other',
                     'CPF': 'Pension Fund',
                     'GPE': 'Pension Fund',
                     'UPE': 'Pension Fund',
                     'RIN': 'Insurance Company',
                     'SBC': 'Investment Manager',
                     'SPZ': 'Investment Manager',
                     'UIT': 'Investment Manager'}
    return dict_type_agg[typecode]


def get_rating_number(rating_class: str):
    lst_rating = ['AAA',
                  'AA+',
                  'AA',
                  'AA-',
                  'A+',
                  'A',
                  'A-',
                  'BBB+',
                  'BBB',
                  'BBB-',
                  'BB+',
                  'BB',
                  'BB-',
                  'B+',
                  'B',
                  'B-',
                  'CCC+',
                  'CCC',
                  'CCC-',
                  'CC',
                  'C',
                  'C-',
                  'D']
    dict_rating = {lst_rating[i]: i for i in range(len(lst_rating))}
    dict_rating['NR'] = np.nan
    return dict_rating[rating_class]


# %%
# Log

def log_quarter(q: pd.Period):
    print('Importing:   ', q)


def log_import_emaxx_holding(df_emaxx_holding: pd.DataFrame):
    print()
    print('Finished import eMAXX holdings')


def log_import_emaxx_fund(df_emaxx_fund: pd.DataFrame):
    print('Finished import eMAXX fund')


def log_import_emaxx_secmast(df_emaxx_fund: pd.DataFrame):
    print('Finished import eMAXX secmast')


def log_import_emaxx_issuer(df_emaxx_issuer: pd.DataFrame):
    print('Finished import eMAXX issuer')


def log_import_emaxx_broktran(df_emaxx_broktran: pd.DataFrame):
    print('Finished import eMAXX broktran')


def log_import_fed(df_fed: pd.DataFrame):
    print()
    print('Imported Federal Reserve treasury holdings')


def log_import_treasury(df_treasury: pd.DataFrame):
    print('Imported Treasury auction data')


def log_import_crsp(df_crsp: pd.DataFrame):
    print('Imported CRSP Treasury price data')


def log_import_wrds_bond(df_wrds_bond: pd.DataFrame):
    print('Imported WRDS bond returns')


def log_clean_emaxx_holding(df_emaxx_holding_clean):
    print()
    print('Cleaned eMAXX holdings')
    print('Number of holdings:  ', len(df_emaxx_holding_clean))


def log_clean_emaxx_fund(df_emaxx_fund_clean: pd.DataFrame):
    print()
    print('Cleaned eMAXX fund characteristics')
    print('Number of investors:  ', df_emaxx_fund_clean['inv_id'].nunique())


def log_clean_emaxx_secmast(df_emaxx_secmast: pd.DataFrame):
    print()
    print('Cleaned eMAXX secmast characteristics')


def log_clean_emaxx_issuer(df_emaxx_issuer: pd.DataFrame):
    print()
    print('Cleaned eMAXX issuer characteristics')
    print('Number of issuers:  ', df_emaxx_issuer['issuer_id'].nunique())


def log_clean_emaxx_broktran(df_emaxx_broktran: pd.DataFrame):
    print()
    print('Cleaned eMAXX broktran')
    print('Number of assets:  ', df_emaxx_broktran['asset_id'].nunique())


def log_clean_fed(df_fed_clean: pd.DataFrame):
    print()
    print('Cleaned Federal Reserve treasury holdings')
    print('Number of holdings:  ', len(df_fed_clean))


def log_clean_treasury(df_treasury_clean: pd.DataFrame):
    print()
    print('Cleaned Treasury auction data')
    print('Number of treasuries:  ', df_treasury_clean['asset_id'].nunique())


def log_clean_crsp(df_crsp_clean: pd.DataFrame):
    print()
    print('Cleaned CRSP prices')
    print('Number of treasuries:  ', df_crsp_clean['asset_id'].nunique())


def log_clean_wrds_bond(df_wrds_bond_clean: pd.DataFrame):
    print()
    print('Cleaned WRDS bond returns')
    print('Number of assets:  ', df_wrds_bond_clean['asset_id'].nunique())


def log_factors(df_factor: pd.DataFrame):
    print()
    print('Calculated treasury factors')
    print('Number of asset/dates:  ', len(df_factor))


def log_corp_treasury(df_asset: pd.DataFrame):
    print('Merged corporate/treasury prices and factors')
    print('Number of asset/dates:  ', len(df_asset))


def log_fed_holding(df_holding_fed: pd.DataFrame):
    print()
    print('Merged Federal Reserve and corporate holdings')


def log_holding_fund_merge(df_holding_fund: pd.DataFrame):
    print('Merged holdings and fund typecode')


def log_holding_factor_merge(df_merged: pd.DataFrame):
    print('Merged holdings and factors')
    print('Number of investor/date/asset:  ', len(df_merged))


def log_drop_unused(df_dropped: pd.DataFrame):
    print()
    print('Dropped unused investors/assets')
    print('Number of holdings:  ', len(df_dropped))


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
    print('Number of non-zero holdings:  ', sum(df_holding['paramt'] > 0))
    print('Number of zero holdings:  ', sum(df_holding['paramt'] == 0))


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


def log_finished(df: pd.DataFrame):
    print()
    print('\n---------------------------Finished---------------------------\n')
    return df


# %%
# Global parameters

sns.set_theme(style='ticks', palette=sns.color_palette('hls', 6), context='paper')

input_path = 'data'
output_path = 'output/'
figure_path = 'figures/'

os.makedirs(output_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

start_date = pd.Period('2017Q3')
end_date = pd.Period('2017Q4')

characteristics = ['coupon',
                   'amtout',
                   't_maturity',
                   # 'is_treasury',
                   'credit_rating',
                   'spread',
                   'const']
params = ['beta_ytm'] + get_param_cols(characteristics)

min_n_holding = 1000
n_quarters = 11
indexfund_id = 22225

# Main

dfs = pandas_read_bond(input_path, start_date, end_date)

df_emaxx_holding_clean, df_emaxx_fund_clean, df_fed_clean, df_treasury_clean, df_crsp_clean, df_wrds_bond_clean = clean_imports_bond(
    *dfs,
    start_date,
    end_date
)

df_asset = (df_crsp_clean
            .pipe(calc_factors, df_treasury_clean, end_date)
            .pipe(merge_corp_treasury, df_wrds_bond_clean))

df_holding = (df_emaxx_holding_clean
              .pipe(merge_holding_fund, df_emaxx_fund_clean)
              .pipe(merge_fed_holding, df_fed_clean)
              .pipe(construct_zero_holdings, n_quarters))

df_results = (df_holding
              .pipe(merge_holding_factor, df_asset)
              .pipe(drop_unused_inv_asset)
              .pipe(create_household_sector)
              .pipe(partition_outside_asset)
              .pipe(calc_inv_aum)
              .pipe(agg_small_inv)
              .pipe(bin_inv, min_n_holding)
              .pipe(calc_instrument)
              .pipe(calc_holding_weights)
              .pipe(estimate_model_L, characteristics, params, min_n_holding)
              .pipe(calc_latent_demand, characteristics, params)
              .pipe(save_df, output_path, 'df_results.csv')
              .pipe(clean_figures)
              .pipe(typecode_share_counts, figure_path)
              .pipe(check_moment_condition, min_n_holding)
              .pipe(critical_value_test, characteristics, min_n_holding, figure_path)
              .pipe(test_index_fund, characteristics, params, indexfund_id, figure_path)
              .pipe(graph_type_params, params, figure_path)
              .pipe(graph_std_latent_demand, figure_path)
              .pipe(log_finished))
