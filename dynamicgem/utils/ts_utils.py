import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D
from time import time
import pandas as pd

def construct_rnn_model(look_back,
                        d,
                        n_units=[20, 20],#[500, 500],
                        dense_units=[50, 10],#[1000, 200, 50, 10],
                        filters=64,
                        kernel_size=5,
                        pool_size=4,
                        method='sgru',
                        bias_reg=None,
                        input_reg=None,
                        recurr_reg=None):
    model = Sequential()
    if method == 'lstm':
        model.add(LSTM(n_units[0],
                       input_shape=(look_back, d),
                       return_sequences=True,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=input_reg,
                       recurrent_regularizer=recurr_reg))
        for n_unit in n_units[1:]:
            model.add(LSTM(n_unit,
                           bias_regularizer=bias_reg,
                           kernel_regularizer=input_reg,
                           recurrent_regularizer=recurr_reg))
    elif method == 'gru':
        model.add(GRU(n_units[0],
                      input_shape=(look_back, d),
                      return_sequences=True,
                      bias_regularizer=bias_reg,
                      kernel_regularizer=input_reg,
                      recurrent_regularizer=recurr_reg))
        for n_unit in n_units[1:]:
            model.add(GRU(n_unit,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg))
    elif method == 'bi-lstm':
        model.add(Bidirectional(LSTM(n_units[0],
                                input_shape=(look_back, d),
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(LSTM(n_unit)))
    elif method == 'bi-gru':
        model.add(Bidirectional(GRU(n_units[0],
                                input_shape=(look_back, d),
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(GRU(n_unit)))
    elif method == 'lstm-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(n_units[0],
                       return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(LSTM(n_unit))
    elif method == 'gru-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(GRU(n_units[0],
                      return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(GRU(n_unit))
    elif method == 'bi-lstm-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Bidirectional(LSTM(n_units[0],
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(LSTM(n_unit))
    elif method == 'bi-gru-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Bidirectional(GRU(n_units[0],
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(GRU(n_unit))
    for dense_n_unit in dense_units:
        model.add(Dense(dense_n_unit, activation='relu'))
    model.add(Dense(d))
    if 'plstm' in method:
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='mean_squared_error', optimizer=adam)
    else:
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# convert an array of values into a dataset matrix
def create_training_samples(graphs, look_back=5, d=2):
    T = len(graphs)
    train_size = T - look_back
    trainX = np.zeros((train_size, look_back, d))
    trainY = np.zeros((train_size, d))
    n_samples_train = 0
    for t in range(T - look_back):
        for tau in range(look_back):
            trainX[n_samples_train, tau, :] = ts_train.iloc[t + tau, :]
        trainY[n_samples_train, :] = ts_train.iloc[t + look_back, :]
        n_samples_train += 1
    return trainX, trainY


def learn_rnn_parameters(ts, ts_exogs, options, look_ahead=1,
                         train_start_date=None,
                         train_end_date=None, test_start_date=None,
                         save_plots=False, method='gru'):
    if train_start_date is None:
        if ts_exogs is None:
            train_start_date = ts.index.min()
        else:
            train_start_date = max(ts.index.min(), ts_exogs.index.min())

    if train_end_date is None:
        train_end_date = max(ts.index)

    test_end_date = options.warn_start_date + pd.Timedelta(days=look_ahead - 1)

    gap_dates = pd.date_range(train_end_date + pd.Timedelta(days=1),
                              test_start_date, closed="left")

    test_dates = pd.date_range(test_start_date, test_end_date)

    gap_and_test_dates = gap_dates.append(test_dates)

    ts_train = ts[train_start_date: train_end_date]

    if ts_exogs is not None:
        ts_exogs_train = ts_exogs[train_start_date: train_end_date]
        ts_exogs_gap_test = ts_exogs[
            min(gap_and_test_dates):max(gap_and_test_dates)]
    else:
        ts_exogs_train = None
        ts_exogs_gap_test = None

    if ts_exogs_train is not None:
        ts_concat = pd.concat([ts_train, ts_exogs_train], axis=1)
    else:
      ts_concat = pd.DataFrame(ts_train)
    ts_concat = ts_concat.dropna(axis=0)
    look_back = 5
    d = len(ts_concat.columns)
    model = construct_rnn_model(look_back=look_back, d=d, method=method)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    trainX, trainY = create_training_samples(
        ts_concat,
        look_back=look_back,
        d=d
    )
    t1 = time()
    model.fit(trainX,
              trainY,
              nb_epoch=2000,
              batch_size=100,
              validation_split=0.2,
              callbacks=[early_stop],
              verbose=2)
    t2 = time()
    print('Training time: %fsec' % (t2 - t1))
    forecast_length = len(gap_dates) + len(test_dates)
    predictions = []
    for pred_day in range(forecast_length):
        testX = np.array(ts_concat[-look_back:]).reshape((1, look_back, d))
        prediction = model.predict(testX, batch_size=100, verbose=0)
        ts_concat = ts_concat.append(pd.DataFrame(prediction, columns=ts_concat.columns))
        predictions.append(prediction[0])
    print('Test time: %fsec' % (time() - t2))
    list_pred = np.array(predictions)[-len(test_dates):, 0]

    print("Last", look_ahead, "predictions:")
    print(list_pred)
    ts_pred = pd.Series(list_pred, index=test_dates + pd.Timedelta(hours=12))
    ts_pred[ts_pred < 0] = 0
    ts_pred.name = 'count'
    ts_pred.index.name = 'date'

    return ts_pred