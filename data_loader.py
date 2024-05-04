import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import dotenv_values

secrets = dotenv_values(".env")


def load_fred_data():
    start_date = '1970-01-01'
    end_date = '2023-11-01'

    fred = Fred(api_key=secrets['API_KEY'])
    inflation_data = pd.DataFrame(fred.get_series('CPIAUCSL', units='pc1', observation_start=start_date,
                                                  observation_end=end_date))
    unemployment_data = pd.DataFrame(fred.get_series('UNRATE', observation_start=start_date, 
                                                     observation_end=end_date))
    ffr_data = pd.DataFrame(fred.get_series('DFF', frequency='m', observation_start=start_date, 
                                            observation_end=end_date))
    market_yield_10_data = pd.DataFrame(fred.get_series('DGS10', frequency='m', observation_start=start_date, 
                                                        observation_end=end_date))
    ppi_data = pd.DataFrame(fred.get_series('PPIACO', frequency='m', units='pc1', observation_start=start_date,
                                            observation_end=end_date))
    gdp_data = pd.DataFrame(fred.get_series('GDPC1', frequency='q', units='pc1', observation_start=start_date, 
                                            observation_end=end_date))
    gdp_invest_data = pd.DataFrame(fred.get_series('GPDIC1', frequency='q', units='pc1', observation_start=start_date, 
                                                   observation_end=end_date))
    sentiment_data = pd.DataFrame(fred.get_series('UMCSENT', units='pc1', observation_start=start_date, 
                                                  observation_end=end_date)).bfill()
    oil_data = pd.DataFrame(fred.get_series('WTISPLC', units='pc1', observation_start=start_date,
                                            observation_end=end_date))
    recession_data = pd.DataFrame(fred.get_series('USREC', observation_start=start_date, 
                                                  observation_end=end_date))

    return inflation_data, unemployment_data, ffr_data, market_yield_10_data, ppi_data, gdp_data, \
           gdp_invest_data, sentiment_data, oil_data, recession_data


def load_yfinance_data():
    sp500_data = yf.download('^GSPC', start='1969-01-01', end='2023-12-01', progress=False)
    sp500_data['Date'] = pd.to_datetime(sp500_data.index)
    sp500_data = sp500_data.set_index('Date')
    sp500_monthly_averages = sp500_data['Adj Close'].resample('ME').mean()
    sp500_monthly_averages = sp500_monthly_averages.reset_index()

    return sp500_monthly_averages


def interpolate_data(data, column_name):
    data = data.reset_index()
    data = data.rename(columns={0: column_name, 'index': 'Date'})
    column_data = pd.DataFrame(data)
    column_data['Date'] = pd.to_datetime(column_data['Date'])
    all_dates = pd.date_range(start=column_data['Date'].min(), end='2023-11-01', freq='MS')
    column_data = pd.merge_asof(pd.DataFrame({'Date': all_dates}), column_data, on='Date', direction='backward')
    column_data[column_name] = column_data[column_name].interpolate()

    return column_data


def create_dataframe():
    inflation_data, unemployment_data, ffr_data, market_yield_10_data, ppi_data, gdp_data, \
    gdp_invest_data, sentiment_data, oil_data, recession_data = load_fred_data()
    sp500_data = load_yfinance_data()

    gdp = interpolate_data(gdp_data, 'GDP')
    gdp_inv = interpolate_data(gdp_invest_data, 'GDP_INV')

    df = inflation_data
    df = df.rename(columns={0: 'inflation_rate'})
    df['unemployment_rate'] = list(unemployment_data[0])
    df['federal_funds_rate'] = list(ffr_data[0])
    df['market_yield_10_rate'] = list(market_yield_10_data[0])
    df['ppi_rate'] = list(ppi_data[0])
    df['gdp_pc1'] = list(gdp['GDP'])
    df['gdp_invest_pc1'] = list(gdp_inv['GDP_INV'])
    df['sp500_yoy'] = list(sp500_data['Adj Close'].pct_change(12).dropna(how='all'))
    df['consumer_senti_pc1'] = list(sentiment_data[0])
    df['oil_price_pc1'] = list(oil_data[0])
    df['recessions'] = list(recession_data[0])

    return df


if __name__ == "__main__":
    df = create_dataframe()
    df.to_csv('Data/original_data.csv', index_label='date')
    print("Dataset Exported")
