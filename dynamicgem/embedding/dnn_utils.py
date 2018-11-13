import numpy as np
import networkx as nx
import pdb
from keras.layers import Input, Dense, LSTM, GRU, Lambda, merge, Reshape
from keras.models import Model, model_from_json, Sequential
import keras.regularizers as Reg
import keras.backend as KBack
from keras.layers.advanced_activations import LeakyReLU


def model_batch_predictor(model, X, batch_size):
    n_samples = X.shape[0]
    counter = 0
    pred = None
    pred2 = None
    while counter < n_samples // batch_size:
        next_pred, curr_pred = \
            model.predict(X[batch_size * counter:batch_size * (counter + 1),
                          :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
        counter += 1
    if n_samples % batch_size != 0:
        next_pred, curr_pred = \
            model.predict(X[batch_size * counter:, :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
    try:
        return pred #, pred2
    except:
        pdb.set_trace()     


def model_batch_predictor_dynrnn(model, graphs, batch_size):
    n_samples = graphs[0].number_of_nodes()
    look_back = len(graphs)
    d = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    counter = 0
    pred = None
    pred2 = None
    while counter < n_samples // batch_size:
        indices = range(batch_size * counter, batch_size * (counter + 1))
        X = np.zeros((batch_size, look_back, d))
        X2 = np.zeros((batch_size, d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau, :] = graphs_sp[tau][node_idx, :].toarray()

        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
        counter += 1
    if n_samples % batch_size != 0:
        indices = range(batch_size * counter, n_samples)
        X = np.zeros((len(indices), look_back, d))
        X2 = np.zeros((len(indices), d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau, :] = graphs_sp[tau][node_idx, :].toarray()
        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
    try:
        # return pred, pred2
        return pred[:,-1,:].reshape((pred.shape[0], pred.shape[2])), pred2
    except:
        pdb.set_trace()

def model_batch_predictor_dynrnn_v2(model, graphs, batch_size):
    n_samples = graphs[0].number_of_nodes()
    look_back = len(graphs)
    d = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    counter = 0
    pred = None
    pred2 = None
    while counter < n_samples // batch_size:
        indices = range(batch_size * counter, batch_size * (counter + 1))
        X = np.zeros((batch_size, look_back, d))
        X2 = np.zeros((batch_size, d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau, :] = graphs_sp[tau][node_idx, :].toarray()

        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
        counter += 1
    if n_samples % batch_size != 0:
        indices = range(batch_size * counter, n_samples)
        X = np.zeros((len(indices), look_back, d))
        X2 = np.zeros((len(indices), d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau, :] = graphs_sp[tau][node_idx, :].toarray()
        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
    try:
        # return pred, pred2
        return pred.reshape((pred.shape[0], pred.shape[2])), pred2
    except:
        pdb.set_trace()


def model_batch_predictor_dynaernn(model, graphs, batch_size):
    n_samples = graphs[0].number_of_nodes()
    look_back = len(graphs)
    n = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    counter = 0
    pred = None
    pred2 = None
    while counter < n_samples // batch_size:
        indices = range(batch_size * counter, batch_size * (counter + 1))
        X = np.zeros((batch_size, look_back * n))
        X2 = np.zeros((batch_size, n))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau*n:(tau+1)*n] = graphs_sp[tau][node_idx, :].toarray()

        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
        counter += 1
    if n_samples % batch_size != 0:
        indices = range(batch_size * counter, n_samples)
        X = np.zeros((len(indices), look_back * n))
        X2 = np.zeros((len(indices), n))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau*n:(tau+1)*n] = graphs_sp[tau][node_idx, :].toarray()
        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
    return pred, pred2



def model_batch_predictor_dynae_v2(model, graphs, batch_size):
    n_samples = graphs[0].number_of_nodes()
    look_back = len(graphs)
    d = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    counter = 0
    pred = None
    pred2 = None
    while counter < n_samples // batch_size:
        indices = range(batch_size * counter, batch_size * (counter + 1))
        X = np.zeros((batch_size, look_back, d))
        X2 = np.zeros((batch_size, d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau, :] = graphs_sp[tau][node_idx, :].toarray()

        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
        counter += 1
    if n_samples % batch_size != 0:
        indices = range(batch_size * counter, n_samples)
        X = np.zeros((len(indices), look_back, d))
        X2 = np.zeros((len(indices), d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            for tau in range(look_back):
                X[idx, tau, :] = graphs_sp[tau][node_idx, :].toarray()
        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
    try:
        # return pred, pred2
        return pred.reshape((pred.shape[0], pred.shape[2])), pred2
    except:
        pdb.set_trace()
        
def model_batch_predictor_dynae(model, graphs, batch_size):
    n_samples = graphs[0].number_of_nodes()
    look_back = len(graphs)
    d = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    counter = 0
    pred = None
    pred2 = None
    while counter < n_samples // batch_size:
        indices = range(batch_size * counter, batch_size * (counter + 1))
        X = np.zeros((batch_size, d))
        X2 = np.zeros((batch_size, d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            try:
                X[idx, :] = graphs_sp[0][node_idx, :].toarray()
            except Exception as e:
                print(e.message)
                pdb.set_trace()    

        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
        counter += 1
    if n_samples % batch_size != 0:

        indices = range(batch_size * counter, n_samples)
        X = np.zeros((len(indices), d))
        X2 = np.zeros((len(indices), d))
        for idx, record_id in enumerate(indices):
            node_idx = record_id
            try:
                X[idx, :] = graphs_sp[0][node_idx, :].toarray()
            except Exception as e:
                print(e.message)
                pdb.set_trace() 
        next_pred, curr_pred = \
            model.predict(X)
        if counter:
            pred = np.vstack((pred, curr_pred))
            pred2 = np.vstack((pred2, next_pred))
        else:
            pred = curr_pred
            pred2 = next_pred
    try:
        return pred, pred2
    except:
        pdb.set_trace()


def batch_generator_dynae(graphs, beta, batch_size, look_back, shuffle):
    # pdb.set_trace()
    T = len(graphs)
    d = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    train_size = T - look_back
    if not train_size:
        return
    number_of_batches = (d * train_size) // batch_size
    if number_of_batches<0:
        pdb.set_trace()
    print('# of batches: %d' % number_of_batches)
    counter = 0
    sample_index = np.arange(d * train_size)
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = np.zeros((batch_size, d))
        X_batch2 = np.zeros((batch_size, d))
        for idx, record_id in enumerate(batch_index):
            graph_idx = record_id // d
            node_idx = record_id % d
            X_batch[idx, :] = graphs_sp[graph_idx][node_idx, :].toarray()
            X_batch2[idx] = graphs_sp[graph_idx+1][node_idx, :].toarray()
        # X_batch = graphs_sp[0][batch_index, :].toarray()
        y_batch = np.ones(X_batch2.shape)
        # y_batch = beta * np.ones(X_batch2.shape)
        # y_batch[X_batch2 == 0] = np.random.choice([0, 1], p=[0.9, 0.1])
        y_batch[X_batch2 != 0] = beta
        y_batch[X_batch2 == 0] = -1
        counter += 1
        # np.savetxt('debug_x_batch.txt',
        #                X_batch.reshape((batch_size, d)))
        # np.savetxt('debug_x_batch2.txt',
        #                X_batch.reshape((batch_size, d)))
        # np.savetxt('debug_y_batch.txt',
        #                y_batch)
        yield [X_batch, X_batch2], y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generator_dynrnn(graphs, beta, batch_size, look_back, shuffle):
    T = len(graphs)
    n = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    train_size = T - look_back
    number_of_batches = (n * train_size) // batch_size
    counter = 0
    sample_index = np.arange(n * train_size)
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = np.zeros((batch_size, look_back, n))
        X_batch2 = np.zeros((batch_size, n))
        for idx, record_id in enumerate(batch_index):
            graph_idx = record_id // n
            node_idx = record_id % n
            # X_batch[idx, :] = graphs_sp[graph_idx][node_idx, :].toarray()
            # X_batch2[idx] = graphs_sp[graph_idx+1][node_idx, :].toarray()
            for tau in range(look_back):
                X_batch[idx, tau, :] = graphs_sp[graph_idx+tau][node_idx, :].toarray()
            X_batch2[idx] = graphs_sp[graph_idx+look_back][node_idx, :].toarray()
        y_batch = beta * np.ones(X_batch2.shape)
        y_batch[X_batch2 != 0] = beta
        y_batch[X_batch2 == 0] = -1
        # y_batch[X_batch2 == 0] = 0
        counter += 1
        # np.savetxt('debug_x_batch.txt',
        #                X_batch.reshape((batch_size, d)))
        # np.savetxt('debug_x_batch2.txt',
        #                X_batch.reshape((batch_size, d)))
        # np.savetxt('debug_y_batch.txt',
        #                y_batch)
        yield [X_batch, X_batch2], y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generator_dynaernn(graphs, beta, batch_size, look_back, shuffle):
    T = len(graphs)
    n = graphs[0].number_of_nodes()
    graphs_sp = []
    for graph in graphs:
        graphs_sp.append(nx.to_scipy_sparse_matrix(graph))
    train_size = T - look_back
    number_of_batches = (n * train_size) // batch_size
    counter = 0
    sample_index = np.arange(n * train_size)
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = np.zeros((batch_size, look_back * n))
        X_batch2 = np.zeros((batch_size, n))
        for idx, record_id in enumerate(batch_index):
            graph_idx = record_id // n
            node_idx = record_id % n
            # X_batch[idx, :] = graphs_sp[graph_idx][node_idx, :].toarray()
            # X_batch2[idx] = graphs_sp[graph_idx+1][node_idx, :].toarray()
            for tau in range(look_back):
                X_batch[idx, tau*n:(tau+1)*n] = graphs_sp[graph_idx+tau][node_idx, :].toarray()
            X_batch2[idx] = graphs_sp[graph_idx+look_back][node_idx, :].toarray()
        y_batch = beta * np.ones(X_batch2.shape)
        y_batch[X_batch2 != 0] = beta
        y_batch[X_batch2 == 0] = -1
        # y_batch[X_batch2 == 0] = 0
        counter += 1
        # np.savetxt('debug_x_batch.txt',
        #                X_batch.reshape((batch_size, d)))
        # np.savetxt('debug_x_batch2.txt',
        #                X_batch.reshape((batch_size, d)))
        # np.savetxt('debug_y_batch.txt',
        #                y_batch)
        yield [X_batch, X_batch2], y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generator_sdne(X, beta, batch_size, shuffle):
    row_indices, col_indices = X.nonzero()
    sample_index = np.arange(row_indices.shape[0])
    number_of_batches = row_indices.shape[0] // batch_size
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch_v_i = X[row_indices[batch_index], :].toarray()
        X_batch_v_j = X[col_indices[batch_index], :].toarray()
        InData = np.append(X_batch_v_i, X_batch_v_j, axis=1)

        B_i = np.ones(X_batch_v_i.shape)
        B_i[X_batch_v_i != 0] = beta
        B_j = np.ones(X_batch_v_j.shape)
        B_j[X_batch_v_j != 0] = beta
        X_ij = X[row_indices[batch_index], col_indices[batch_index]]
        deg_i = np.sum(X_batch_v_i != 0, 1).reshape((batch_size, 1))
        deg_j = np.sum(X_batch_v_j != 0, 1).reshape((batch_size, 1))
        a1 = np.append(B_i, deg_i, axis=1)
        a2 = np.append(B_j, deg_j, axis=1)
        OutData = [a1, a2, X_ij.T]
        counter += 1
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def get_encoder(node_num, d, n_units, nu1, nu2, activation_fn):
    K = len(n_units) + 1
    # Input
    x = Input(shape=(node_num,))
    # Encoder layers
    y = [None] * (K + 1)
    y[0] = x  # y[0] is assigned the input
    for i in range(K - 1):
        y[i + 1] = Dense(n_units[i], activation=activation_fn,
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[i])
    y[K] = Dense(d, activation=activation_fn,
                 W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[K - 1])
    # Encoder model
    encoder = Model(input=x, output=y[K])
    return encoder

def get_encoder_dynaernn(node_num, d, n_units, nu1, nu2, activation_fn):
    K = len(n_units) + 1
    # Input
    x = Input(shape=(node_num,))
    # Encoder layers
    y = [None] * (K + 1)
    y[0] = x  # y[0] is assigned the input
    for i in range(K - 1):
        y[i + 1] = Dense(n_units[i], activation=LeakyReLU(),
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[i])
    y[K] = Dense(d, activation=LeakyReLU(),
                 W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y[K - 1])
    # Encoder model
    encoder = Model(input=x, output=y[K])
    return encoder


def get_decoder(node_num, d,
                n_units, nu1, nu2,
                activation_fn):
    K = len(n_units) + 1
    # Input
    y = Input(shape=(d,))
    # Decoder layers
    y_hat = [None] * (K + 1)
    y_hat[K] = y
    for i in range(K - 1, 0, -1):
        y_hat[i] = Dense(n_units[i - 1],
                         activation=activation_fn,
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i + 1])
    y_hat[0] = Dense(node_num, activation=activation_fn,
                     W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1])
    # Output
    x_hat = y_hat[0]  # decoder's output is also the actual output
    # Decoder Model
    decoder = Model(input=y, output=x_hat)
    return decoder

def get_decoder_dynaernn(node_num, d,
                n_units, nu1, nu2,
                activation_fn):
    K = len(n_units) + 1
    # Input
    y = Input(shape=(d,))
    # Decoder layers
    y_hat = [None] * (K + 1)
    y_hat[K] = y
    for i in range(K - 1, 0, -1):
        y_hat[i] = Dense(n_units[i - 1],
                         activation=LeakyReLU(),
                         W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i + 1])
    y_hat[0] = Dense(node_num, activation=LeakyReLU(),
                     W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1])

    # Output
    x_hat = y_hat[0]  # decoder's output is also the actual output
    # Decoder Model
    decoder = Model(input=y, output=x_hat)
    return decoder    


def get_autoencoder(encoder, decoder):
    # Input
    x = Input(shape=(encoder.layers[0].input_shape[1],))
    # Generate embedding
    y = encoder(x)
    # Generate reconstruction
    x_hat = decoder(y)
    # Autoencoder Model
    autoencoder = Model(input=x, output=[x_hat, y])
    return autoencoder

def get_lstm_encoder(n_nodes, look_back, d,
                n_units, activation_fn,
                bias_reg, input_reg, recurr_reg,
                ret_seq=True
                ):
    model = Sequential()
    # model.add(Dense(d, input_shape=(look_back, n_nodes),
    #             activation=activation_fn,
    #              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
    #              )
    n_rnn_layers = len(n_units)
    return_sequences = bool(n_rnn_layers - 1)
    model.add(LSTM(n_units[0],
                   input_shape=(look_back, n_nodes),
                   return_sequences=return_sequences,
                   bias_regularizer=bias_reg,
                   kernel_regularizer=input_reg,
                   recurrent_regularizer=recurr_reg
                   

                   )
    )
    for l_idx, n_unit in enumerate(n_units[1:-1]):
        model.add(LSTM(n_unit,
                       return_sequences=True,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=input_reg,
                       recurrent_regularizer=recurr_reg
                       )
                  )
    if n_rnn_layers > 1:
        model.add(LSTM(n_units[-1],
                           return_sequences=False,
                           bias_regularizer=bias_reg,
                           kernel_regularizer=input_reg,
                           recurrent_regularizer=recurr_reg
                           )
                      )
    return model


def get_lstm_decoder(n_nodes, look_back, d,
                n_units, activation_fn,
                nu1, nu2,
                bias_reg, input_reg, recurr_reg
                ):
    model = Sequential()
    n_rnn_layers = len(n_units)
    # model.add(LSTM(d,
    #                input_shape=(look_back, d),
    #                return_sequences=True,
    #                bias_regularizer=bias_reg,
    #                kernel_regularizer=input_reg,
    #                recurrent_regularizer=recurr_reg
    #                )
    # )
    for l_idx, n_unit in enumerate(n_units[::-1]):
        if l_idx < n_rnn_layers - 1:
            model.add(LSTM(n_unit,
                          return_sequences=True,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg
                          )
                      )
        else:
            model.add(LSTM(n_nodes,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg
                          )
                      )
            
    # model.add(Dense(n_nodes, activation=activation_fn,
    #              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
    #              )
    return model

def get_lstm_encoder_v2(n_nodes, look_back, d,
                n_units, activation_fn, nu1,nu2,
                bias_reg, input_reg, recurr_reg,
                ret_seq=True
                ):
    model = Sequential()
    # model.add(Dense(d, input_shape=(look_back, n_nodes),
    #             activation=activation_fn,
    #              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
    #              )
    n_rnn_layers = len(n_units)
    model.add(LSTM(n_units[0],
                   input_shape=(look_back, n_nodes),
                   return_sequences=True,
                   bias_regularizer=bias_reg,
                   kernel_regularizer=input_reg,
                   recurrent_regularizer=recurr_reg
                   

                   )
    )
    for l_idx, n_unit in enumerate(n_units[1:]):
        model.add(LSTM(n_unit,
                       return_sequences=True,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=input_reg,
                       recurrent_regularizer=recurr_reg
                       )
                  )
    model.add(LSTM(d,
                       return_sequences=True,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=input_reg,
                       recurrent_regularizer=recurr_reg
                       )
                  )            
    return model

def get_lstm_decoder_v2(n_nodes, look_back, d,
                n_units, activation_fn,
                nu1, nu2,
                bias_reg, input_reg, recurr_reg
                ):
    model = Sequential()
    n_rnn_layers = len(n_units)
    model.add(LSTM(d,
                   input_shape=(look_back, d),
                   # input_shape=(1, n_nodes),
                   return_sequences=True,
                   bias_regularizer=bias_reg,
                   kernel_regularizer=input_reg,
                   recurrent_regularizer=recurr_reg
                   )
    )
    for l_idx, n_unit in enumerate(n_units[::-1]):
        if l_idx < n_rnn_layers - 1:
            model.add(LSTM(n_unit,
                          return_sequences=True,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg
                          )
                      )
        else:
            model.add(LSTM(n_nodes,
                          return_sequences=False,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg
                          )
                      )
            
    # model.add(Dense(n_nodes, activation=activation_fn,
    #              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
    #              )
    return model


def get_lstm_encoder_v3(n_nodes, look_back, d,
                n_units, activation_fn, nu1,nu2,
                bias_reg, input_reg, recurr_reg,
                ret_seq=True
                ):
    model = Sequential()
    # model.add(Dense(d, input_shape=(look_back, n_nodes),
    #             activation=activation_fn,
    #              W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
    #              )
    n_rnn_layers = len(n_units)
    model.add(LSTM(n_units[0],
                   input_shape=(look_back, n_nodes),
                   return_sequences=True,
                   bias_regularizer=bias_reg,
                   kernel_regularizer=input_reg,
                   recurrent_regularizer=recurr_reg
                   

                   )
    )
    for l_idx, n_unit in enumerate(n_units[1:-1]):
        model.add(LSTM(n_unit,
                       return_sequences=True,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=input_reg,
                       recurrent_regularizer=recurr_reg
                       )
                  )
    if n_rnn_layers > 1:
        model.add(LSTM(n_units[-1],
                           return_sequences=False,
                           bias_regularizer=bias_reg,
                           kernel_regularizer=input_reg,
                           recurrent_regularizer=recurr_reg
                           )
                      )
    # model.add(Reshape((1, d)))                        
    model.add(Dense(d, activation=activation_fn,
                 W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
                 )    
    # model.add(LSTM(d,
    #                return_sequences=False,
    #                bias_regularizer=bias_reg,
    #                kernel_regularizer=input_reg,
    #                recurrent_regularizer=recurr_reg
    #                )
    #               )  
    model.add(Reshape((1, d)))                      
    return model

def get_lstm_decoder_v3(n_nodes, look_back, d,
                n_units, activation_fn,
                nu1, nu2,
                bias_reg, input_reg, recurr_reg
                ):
    model = Sequential()
    n_rnn_layers = len(n_units)
    model.add(LSTM(d,
                   input_shape=(1,d),
                   # input_shape=(1, n_nodes),
                   return_sequences=True,
                   bias_regularizer=bias_reg,
                   kernel_regularizer=input_reg,
                   recurrent_regularizer=recurr_reg
                   )
    )
    for l_idx, n_unit in enumerate(n_units[::-1]):
        if l_idx < n_rnn_layers - 1:
            model.add(LSTM(n_unit,
                          return_sequences=True,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg
                          )
                      )
        else:
            model.add(LSTM(n_nodes,
                          return_sequences=False,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg
                          )
                      )
            
    model.add(Dense(n_nodes, activation=activation_fn,
                 W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))
                 )
    return model

def get_lstm_autoencoder_v2(encoder, decoder,d):
    # Input
    x = Input(shape=(encoder.layers[0].input_shape[1], encoder.layers[0].input_shape[2]))
    # Generate embedding
    try:
        y = encoder(x)
    except:
        pdb.set_trace()
    # Generate reconstruction
    try:
        # y=KBack.reshape(y,(-1,1,d))
        x_hat = decoder(y)
    except:
        pdb.set_trace()        
    
        
    # Autoencoder Model
    autoencoder = Model(input=x, output=[x_hat, y])
    return autoencoder    

# def get_lstm_decoder_v3(n_nodes, look_back, d,
#                 n_units, activation_fn,
#                 nu1, nu2,
#                 bias_reg, input_reg, recurr_reg
#                 ):
#     K = len(n_units) + 1
#     # Input
#     y = Input(shape=(d,))
#     # Decoder layers
#     y_hat = [None] * (K + 1)
#     y_hat[K] = y
#     for i in range(K - 1, 0, -1):
#         y_hat[i] = Dense(n_units[i - 1],
#                          activation=LeakyReLU(),
#                          W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[i + 1])
#     y_hat[0] = Dense(n_nodes, activation=LeakyReLU(),
#                      W_regularizer=Reg.l1_l2(l1=nu1, l2=nu2))(y_hat[1])

#     # Output
#     x_hat = y_hat[0]  # decoder's output is also the actual output
#     # Decoder Model
#     decoder = Model(input=y, output=x_hat)

#     return decoder    


def get_lstm_autoencoder(encoder, decoder):
    # Input
    x = Input(shape=(encoder.layers[0].input_shape[1], encoder.layers[0].input_shape[2]))
    # Generate embedding
    y = encoder(x)
    # Generate reconstruction
    x_hat = decoder(y)
    # Autoencoder Model
    autoencoder = Model(inputs=x, outputs=[x_hat, y])
    return autoencoder


def get_aelstm_autoencoder(ae_encoders, lstm_encoder, ae_decoder):
    y_enc = [None] * len(ae_encoders)
    inp_size = sum([encoder.layers[0].input_shape[1] for encoder in ae_encoders])
    # Input
    x_in = Input(shape=(inp_size,))
    for enc_idx, ae_enc in enumerate(ae_encoders):
        ae_inp_size = ae_encoders[enc_idx].layers[0].input_shape[1]
        x_i = Lambda(
            lambda x: x[:, enc_idx*ae_inp_size:(enc_idx+1)*ae_inp_size]
        )(x_in)
        y_enc[enc_idx] = ae_encoders[enc_idx](x_i)

    # Ravel AE output for LSTM input
    try:
        y_enc_flat = Lambda(lambda x: KBack.stack(x, axis=1))(y_enc)
    except TypeError: # If look_back = 1
        y_enc_flat = Lambda(lambda x: KBack.reshape(x, (-1, 1, y_enc[0].shape[1])))(y_enc[0])
    # y_enc_flat = KBack.stack(y_enc, axis=1)
    # Generate embedding

    y = lstm_encoder(y_enc_flat)
    # Generate reconstruction
    x_hat = ae_decoder(y)
    # Autoencoder Model
    autoencoder = Model(input=x_in, output=[x_hat, y])
    return autoencoder


def graphify(reconstruction):
    [n1, n2] = reconstruction.shape
    n = min(n1, n2)
    reconstruction = np.copy(reconstruction[0:n, 0:n])
    reconstruction = (reconstruction + reconstruction.T) / 2
    reconstruction -= np.diag(np.diag(reconstruction))
    return reconstruction
    return reconstruction


def loadmodel(filename):
    try:
        model = model_from_json(open(filename).read())
    except:
        print('Error reading file: {0}. Cannot load previous model'.format(filename))
        exit()
    return model


def loadweights(model, filename):
    try:
        model.load_weights(filename)
    except:
        print('Error reading file: {0}. Cannot load previous weights'.format(filename))
        exit()


def savemodel(model, filename):
    json_string = model.to_json()
    open(filename, 'w').write(json_string)


def saveweights(model, filename):
    model.save_weights(filename, overwrite=True)
