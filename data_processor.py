import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def data_scaler(X_train, X_test, y_train, y_test):
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    X_train_scaled = scaler_features.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler_features.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train_scaled = scaler_target.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_target.transform(y_test.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_scaled, \
           y_test_scaled, scaler_target


def ins_data_preprocess(dataset, target_variable, split, time_steps):
    features = dataset.drop(target_variable, axis=1).values
    target = dataset[target_variable].values.reshape(-1, 1)

    X, y = [], []

    for i in range(len(features) - time_steps):
        X.append(np.concatenate((target[i:(i + time_steps)],
                                 features[i:(i + time_steps), :]), axis=1))
        y.append(target[i + time_steps, 0])

    X, y = np.array(X), np.array(y)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test


def ins_data_preprocess_cv(dataset, target_variable, time_steps):
    features = dataset.drop(target_variable, axis=1).values
    target = dataset[target_variable].values.reshape(-1, 1)

    X, y = [], []

    for i in range(len(features) - time_steps):
        X.append(np.concatenate((target[i:(i + time_steps)],
                                 features[i:(i + time_steps), :]), axis=1))
        y.append(target[i + time_steps, 0])

    X, y = np.array(X), np.array(y)

    return X, y


def os_data_preprocess(dataset, target_variable, n_lookback, n_forecast):
    os_y = dataset[target_variable]
    os_y = os_y.values.reshape(-1, 1)

    os_X_features = dataset.drop(target_variable, axis=1).values

    scaler_os_y = MinMaxScaler()
    os_y = scaler_os_y.fit_transform(os_y)

    scaler_os_X = MinMaxScaler()
    os_X_features = scaler_os_X.fit_transform(os_X_features)

    os_X = []
    os_Y = []

    for i in range(n_lookback, len(os_y) - n_forecast + 1):
        os_X.append(np.hstack((os_y[i - n_lookback: i], os_X_features[i - n_lookback: i])))
        os_Y.append(os_y[i: i + n_forecast])

    os_X = np.array(os_X)
    os_Y = np.array(os_Y)

    return os_X, os_Y, os_y, os_X_features, scaler_os_y


def forecast_dataset_creator(dataset, model_Y, upper_bound, lower_bound, n_forecast):
    df_past = dataset[['inflation_rate']].reset_index()
    df_past = df_past.rename(columns={'date': 'Date', 'inflation_rate': 'Actual'})
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past.loc[df_past.index[-1], 'Forecast'] = df_past.loc[df_past.index[-1], 'Actual']

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.DateOffset(months=1), periods=n_forecast, freq='ME')
    df_future['Date'] = df_future['Date'].apply(lambda dt: dt.replace(day=1))
    df_future['Forecast'] = model_Y.flatten()
    df_future['Upper Bound'] = np.mean(upper_bound.squeeze(), axis=1)
    df_future['Lower Bound'] = np.mean(lower_bound.squeeze(), axis=1)
    df_future['Actual'] = np.nan

    results = pd.concat([df_past, df_future]).set_index('Date')
    results = results.astype('float')

    results.loc[results.index[-n_forecast - 1], 'Upper Bound'] = dataset.loc[dataset.index[-1], 'inflation_rate']
    results.loc[results.index[-n_forecast - 1], 'Lower Bound'] = dataset.loc[dataset.index[-1], 'inflation_rate']

    return results


def forecast_dataset_cleaner(out_of_sample_lists, hmm, n_forecast):
    forecast_df = pd.DataFrame()
    forecast_df = out_of_sample_lists[0]
    forecast_df = forecast_df.rename(columns={'Actual': 'actual_rate', 
                                              'Forecast': 'original_forecast',
                                              'Upper Bound': 'original_forecast_u', 
                                              'Lower Bound': 'original_forecast_l'})

    forecast_df['hidden_forecast'] = out_of_sample_lists[1]['Forecast']
    forecast_df['hidden_forecast_u'] = out_of_sample_lists[1]['Upper Bound']
    forecast_df['hidden_forecast_l'] = out_of_sample_lists[1]['Lower Bound']
    forecast_df['means_forecast'] = out_of_sample_lists[2]['Forecast']
    forecast_df['means_forecast_u'] = out_of_sample_lists[2]['Upper Bound']
    forecast_df['means_forecast_l'] = out_of_sample_lists[2]['Lower Bound']
    forecast_df['all_hmm_forecast'] = out_of_sample_lists[3]['Forecast']
    forecast_df['all_hmm_forecast_u'] = out_of_sample_lists[3]['Upper Bound']
    forecast_df['all_hmm_forecast_l'] = out_of_sample_lists[3]['Lower Bound']

    forecast_df = forecast_df.astype('float')

    future_data, hidden_states = hmm.sample(n_forecast)

    forecast_df['hmm_sample_forecast'] = np.nan
    forecast_df['hidden_states_os'] = np.nan

    forecast_df.loc[forecast_df.index[-n_forecast:], 'hmm_sample_forecast'] = future_data[:, 0]
    forecast_df.loc[forecast_df.index[-n_forecast:], 'hidden_states_os'] = hidden_states

    forecast_df.loc[forecast_df.index[-n_forecast - 1], 'hmm_sample_forecast'] = 3.13948
    forecast_df.loc[forecast_df.index[-n_forecast - 1], 'hidden_states_os'] = 0

    return forecast_df
