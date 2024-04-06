import json
import requests
import pandas as pd


def last_wednesday_of_quarter(q: pd.Period):
    end_of_quarter = q.to_timestamp(how='end')

    if end_of_quarter.weekday() >= 2:
        last_wednesday = end_of_quarter - pd.Timedelta(days=(end_of_quarter.weekday() - 2))
    else:
        last_wednesday = end_of_quarter - pd.Timedelta(days=(7 + end_of_quarter.weekday() - 2))

    return last_wednesday.strftime('%Y-%m-%d')


def call_api(as_of_date: str):
    url = f'https://markets.newyorkfed.org/api/soma/tsy/get/asof/{as_of_date}.json'
    response = requests.get(url)
    return json.loads(response.text)['soma']['holdings']


# Federal Reserve Market Data API to compile treasury holdings data
# System Open Market Account Holdings of Domestic Securities
# Documentation: https://markets.newyorkfed.org/static/docs/markets-api.html

if __name__ == '__main__':
    start_date = pd.Period('2003Q3')
    end_date = pd.Period('2023Q4')
    period_range = pd.period_range(start_date, end_date, freq='Q')

    cols = ['cusip',
            'maturityDate',
            'coupon',
            'parValue',
            'percentOutstanding']
    df = pd.DataFrame(columns=cols)

    for q in period_range:
        as_of_date = last_wednesday_of_quarter(q)
        data = call_api(as_of_date)
        df_q = pd.DataFrame(data=data, columns=cols).assign(date=q)
        df = pd.concat([df, df_q])
        print('Imported  ', q)

    print('Finished compiling SOMA data')
    print()
    print(df.info(verbose=True))
    df.to_csv('fed_treasury_holding.csv', index=False)
