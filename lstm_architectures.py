import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def integrated_gradients(model, input_seq, steps=50):
    baseline = np.zeros_like(input_seq)
    scaled_inputs = [baseline + (float(i) / steps) * (input_seq - baseline) for i in range(0, steps + 1)]
    gradients = []

    for scaled_input in scaled_inputs:
        scaled_input = tf.convert_to_tensor([scaled_input], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(scaled_input)
            predictions = model(scaled_input)
            
        grads = tape.gradient(predictions, scaled_input)[0]
        gradients.append(grads)
    
    avg_grads = np.mean(gradients, axis=0)
    integrated_grad = (input_seq - baseline) * avg_grads
    
    return integrated_grad


def in_sample_model(X):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def out_sample_model(X, n_lookback, n_forecast):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, X.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=n_forecast))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def eval_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return (f"Mean Squared Error: {mse.round(3)}\n"
            f"Mean Absolute Error: {mae.round(3)}\n"
            f"R²: {r2.round(3)}\n")
