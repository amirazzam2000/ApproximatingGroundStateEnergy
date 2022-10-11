import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import math
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit


def printProgressBar(
        iteration,
        total,
        prefix='',
        suffix='',
        decimals=1,
        length=100,
        fill='█',
        printEnd="\r"):
    '''
    This function creates a progress bar. 

    '''

    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


class AdamOptim():
    '''
    This class builds an adam optimizer to update the wieghts of the model. 
    '''
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w: csr_matrix, dw: csr_matrix, b=None, db=None, has_bais=False):
        '''
        this method takes the gradient of the wieghts (and biases if applicable) 
        and uses them to update the specified weights using the adam optimization algorithm.
        '''

        self.m_dw = self.beta1*self.m_dw + \
            (dw.multiply(1-self.beta1)).toarray()
        if has_bais:
            self.m_db = self.beta1*self.m_db + \
                (db.multiply(1-self.beta1)).toarray()

        
        self.v_dw = self.beta2*self.v_dw + \
            ((dw.multiply(dw)).multiply(1-self.beta2)).toarray()

        if has_bais:
            self.v_db = self.beta2*self.v_db + \
                (db.multiply(1-self.beta2)).toarray()

        m_dw_corr = self.m_dw/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)

        if has_bais:
            m_db_corr = self.m_db/(1-self.beta1**t)
            v_db_corr = self.v_db/(1-self.beta2**t)

        aux = self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        w = csr_matrix(w.toarray() - aux)
        if has_bais:
            aux = self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
            b = csr_matrix(b.toarray() - aux)
            return w, b

        return w



def sigmoid(x):
    '''
    This function defines a sigmoid activation 
    '''
    return 1 / (1 + math.exp(-x))


def sparse_sig(m: csr_matrix):
    '''
    This function defines a sigmoid activation, but using sparse matrices
    '''
    for x, y in list(zip(m.nonzero())):
        m[x, y] = sigmoid(m[x, y])

def dsigmoid(matrix: csr_matrix):
    '''
    This function defines the derivative of a sigmoid function. 
    '''
    sigmoid_matrix = csr_matrix(expit(matrix.todense()))
    return csr_matrix(
        sigmoid_matrix.multiply(np.subtract(1, sigmoid_matrix.todense()))
    )


def softmax_grad(s):
    '''
    This function defines the gradient of the softmax
    '''
    return np.diagflat(s) - np.dot(s, s.T)


def normal(size):
    '''
    this function returns a numpy array of randomly generated numbers. 
    These numbers are generated according to a normal distribution. 
    
    '''
    return np.random.normal(loc=0, scale=1, size=size)


class LSTM:
    '''
    This class defines the LSTM cell. 
    '''

    def __init__(self, units, num_features):
        '''
        Constructs an LSTM cell.

        :param units: The number of hiddent units in the cell
        :param num_features: The number of features in the input of this cell
        '''
        self.units = units
        self.num_features = num_features

        self.Wadam = AdamOptim()
        self.badam = AdamOptim()
        self.Uadam = AdamOptim()

        self.W = sparse.random(num_features,  # + units,
                               units * 4, density=0.8, data_rvs=normal).tocsr()
        self.U = sparse.random(
            units, units * 4, density=0.5, data_rvs=normal).tocsr()
        self.b = csr_matrix((1, units * 4))

        self.cache = []  # np.memmap('cache.dat')

    def slice_weight(self, M, num_slices, axis=1):
        '''
        This method is used to devied the wieghts into different slices. 
        '''
        out = []

        for i in range(num_slices):
            if axis == 1:
                out.append(M[:, self.units*i: self.units*(i+1)])
            elif axis == 0:
                out.append(M[self.units*i: self.units*(i+1)])

        return out

    def load_pickle(self, lstm):
        '''
        This method is used to initialize the weights from an old model saved in a pickle file
        '''
        self.units = lstm.units
        self.num_features = lstm.num_features

        self.Wadam = lstm.Wadam
        self.badam = lstm.badam
        self.Uadam = lstm.Uadam

        self.W = lstm.W
        self.U = lstm.U
        self.b = lstm.b

        self.cache = []

    def forward(self, x, state, training=True):
        '''
        This method performs the forward propagation. 
        The initial state and the input vector are passed to this method. 
        The method returns the output state of the node. 
        '''
        h_old, c_old = state[0], state[1]

        #x = x.multiply(1000)
        #x = csr_matrix(hstack([x, h_old]))

        wf, wi, wo, wc = self.slice_weight(self.W, 4)
        uf, ui, uo, uc = self.slice_weight(self.U, 4)
        bf, bi, bo, bc = self.slice_weight(self.b, 4)

        f = csr_matrix(expit(csr_matrix(x @ wf + h_old @ uf + bf).todense()))
        i = csr_matrix(expit(csr_matrix(x @ wi + h_old @ ui + bi).todense()))
        o = csr_matrix(expit(csr_matrix(x @ wo + h_old @ uo + bo).todense()))
        c_hat = csr_matrix((x @ wc + h_old @ uc + bc)).tanh()

        c_new = csr_matrix(f.multiply(c_old) + i.multiply(c_hat))
        h_new = csr_matrix(o.multiply(c_new.tanh()))

        state = (h_new, c_new)

        if training:
            dect = {'gates': [f, i, o, c_hat],
                    'c': state[1],
                    'old_state': (h_old, c_old),

                    }
            self.cache.append(dect)

        return state

    def backward(self, dy, x, d_future, return_input=False):
        '''
        This method defines the backward propagation. 
        The gradients of the output are given, 
        and then this method returns gradient of the states and the weights of the LSTM cell. 
        If the parameter return_input is True, then also the gradient of the input is returned. 
        '''
        dect = self.cache.pop()

        f, i, o, c_hat = dect['gates']
        c = dect['c']
        h_old, c_old = dect['old_state']

        #x = csr_matrix(x).multiply(1000)

        #x = csr_matrix(hstack([x, h_old]))

        c_old = csr_matrix(c_old)
        h_old = csr_matrix(h_old)

        uf, ui, uo, uc = self.slice_weight(self.U, 4)
        wf, wi, wo, wc = self.slice_weight(self.W, 4)

        h_future, c_future = d_future

        #print(dy.shape, h_future.shape)
        dh = csr_matrix(dy + h_future).tanh()

        tanh_c = csr_matrix(c.tanh())
        dtanh_c = csr_matrix(np.subtract(1, tanh_c.multiply(tanh_c).todense()))

        do = (csr_matrix(c.tanh()).multiply(dh))
        do = dsigmoid(o).multiply(do)

        dc = (o.multiply(dh).multiply(dtanh_c)) + c_future

        df = (c_old.multiply(dc))
        df = dsigmoid(f).multiply(df)

        di = (c_hat.multiply(dc))
        di = dsigmoid(i).multiply(di)

        tanh_c_hat = csr_matrix(c_hat.tanh())
        dtanh_c_hat = csr_matrix(np.subtract(
            1, tanh_c_hat.multiply(tanh_c_hat).todense()))

        dc_hat = dtanh_c_hat.multiply((i.multiply(dc)))

        dwf = x.transpose() @ df
        duf = df.transpose() @ h_old
        dbf = df

        dwi = x.transpose() @ di
        dui = di.transpose() @ h_old
        dbi = di

        dwo = x.transpose() @ do
        duo = do.transpose() @ h_old
        dbo = do

        dwc = x.transpose() @ dc_hat
        duc = dc_hat.transpose() @ h_old
        dbc = dc_hat

        dW = csr_matrix(hstack([csr_matrix(dwf), dwi, dwo, dwc]))
        dU = csr_matrix(hstack([csr_matrix(duf), dui, duo, duc]))
        db = csr_matrix(hstack([csr_matrix(dbf), dbi, dbo, dbc]))

        dh = df @ uf.transpose() + di @ ui.transpose() + \
            do @ uo.transpose() + dc_hat @ uc.transpose()

        dx = df @ wf.transpose() + di @ wi.transpose() + \
            do @ wo.transpose() + dc_hat @ wc.transpose()

        dh_new = dh

        dc_new = f.multiply(dc)

        grad = {'dW': dW, 'dU': dU, 'db': db}

        state = (dh_new, dc_new)

        if return_input:
            return grad, state, dh, dx
        return grad, state, dh

    def update_wieghts(self, grads, learning_rate, add_noise=False, noise_density=0.15):
        '''
        Updates the weights of the cell using the Adam optimizer. 
        '''
        b = self.b
        self.W = self.Wadam.update(learning_rate, self.W, grads['dW'] + (0 if not add_noise else sparse.random(
            grads['dW'].shape[0], grads['dW'].shape[1], density=noise_density, format='csr', data_rvs=normal)))
        self.U = self.Uadam.update(learning_rate, self.U, grads['dU'] + (0 if not add_noise else sparse.random(
            grads['dU'].shape[0], grads['dU'].shape[1], density=noise_density, format='csr', data_rvs=normal)))
        self.b = self.badam.update(learning_rate, self.b, grads['db'])

    def get_empty_grad_list(self):
        '''
        returns a dictionary with empty sparse matrices that have the same shape as the weights of the network.
        '''
        return {'dW': csr_matrix(self.W.shape),
                'dU': csr_matrix(self.U.shape),
                'db': csr_matrix(self.b.shape),
                }


class Dense:
    '''
    This class defines a dense layer
    '''

    def __init__(self, units):
        '''
        Constructs a dense layer 
        :param units: the number of hidden units in the dense layer
        '''
        self.units = units
        self.Wadam = AdamOptim()
        self.badam = AdamOptim()

        self.Wy = sparse.random(units, 1, density=0.9, data_rvs=normal).tocsr()
        self.by = csr_matrix((1, 1))

        self.cache = []

    def load_pickle(self, dense):
        '''
        initializes the wieghts of the layer with the wieghts of an old model. 
        '''
        self.units = dense.units
        self.Wadam = dense.Wadam
        self.badam = dense.badam
        self.Wy = dense.Wy
        self.by = dense.by
        self.cache = []

    def forward_step(self, x_train, batch_size, training=True):
        '''
        This method preforms the forward propagation of the network. 
        It returns the dense output from the input passed to this layer. 
        '''
        y = []
        for x in x_train:
            pred = csr_matrix(x @ self.Wy)  # + self.by)
            y.append(float(pred[0, 0]))

        return y

    def backward_step(self, x_in, y_pred, y_real, learning_rate=0.01):
        '''
        This method defines the backward propagation. 
        The gradients of the output are given, 
        and then this method returns gradient of the states and the weights of the LSTM cell. 
        If the parameter return_input is True, then also the gradient of the input is returned. 
        '''
        dx = []
        grads = self.get_empty_grad_list()

        for i, (pred, real) in enumerate((list(zip(y_pred, y_real)))):

            dy = pred - real
            dWy = csr_matrix(x_in[i].transpose() @ [[dy]])
            dby = csr_matrix([[dy]])
            dx.append(csr_matrix([[dy]] @ self.Wy.transpose()))
            grad = {'dWy': dWy, 'dby': dby}

            self.update_wieghts(grad, learning_rate)

        return dx

    def update_wieghts(self, grads, learning_rate, add_noise=False, noise_density=0.15):
        '''
        Updates the weights of the cell using the Adam optimizer. 
        '''
        self.Wy = self.Wadam.update(
            learning_rate, self.Wy, grads['dWy'], self.by, grads['dby'], has_bais=False)

    def get_empty_grad_list(self):
        '''
        returns a dictionary with empty sparse matrices that have the same shape as the weights of the network.
        '''
        return {'dWy': csr_matrix(self.Wy.shape),
                'dby': 0,
                }


class LSTM_NN:
    '''
    This class uses the LSTM cell to define an LSTM layer. 
    '''

    def __init__(self, units, num_features):
        '''
        Constructs an LSTM cell

        :param units: The number of hiddent units in the layer
        :param num_features: The number of features in the input of this model
        '''
        self.node = LSTM(units=units, num_features=num_features)

    def load_pickle(self, lstm_nn):
        '''
        This method is used to initialize the weights from an old model saved in a pickle file
        '''
        self.node.load_pickle(lstm_nn.node)

    def forward_step(self, batch_size, X_train, inital_state, verbose=1, return_sequnece=False):
        '''
        This method performs the forward propagation. 
        The initial state and the input vector are passed to this method. 
        The method returns the output state of the node. 

        The return_sequence paramater instructs the network return the full sequence of states instead of the last state only.
        '''
        output = []

        for batch in range(batch_size):
            if verbose == 1:
                i = 0
                try:
                    input_length = len(X_train[batch])
                except:
                    input_length = (X_train[batch]).shape[0]
                printProgressBar(0,
                                 input_length,
                                 prefix='Progress:',
                                 suffix='Complete',
                                 length=50)
            state = inital_state
            aux = []
            for x in X_train[batch]:
                state = self.node.forward(csr_matrix(x), state)
                if verbose == 1:
                    i += 1
                    printProgressBar(i,
                                     input_length,
                                     prefix='Progress:',
                                     suffix='Complete', length=50
                                     )
                if return_sequnece:
                    aux.append(state[0].todense().reshape(state[0].shape[1], ))
            if return_sequnece:
                output.append(csr_matrix(np.array(aux).reshape(
                    np.array(aux).shape[0], np.array(aux).shape[2])))
            else:
                output.append(state[0])

        return output

    def backward_step(self, batch_size, X_train, dPred, inital_state, learning_rate=0.01, verbose=1, add_noise=False):
        '''
        This method defines the backward propagation. 
        The gradients of the output are given, 
        and then this method returns gradient of the states and the weights of the LSTM cell. 
        The add noise parameter adds some noise to the gradient before updating the wieghts of the cell. 
        '''
        dInput = []
        for batch in range(batch_size):

            dy = dPred.pop()
            state = inital_state
            grads = self.node.get_empty_grad_list()

            if verbose == 1:
                try:
                    input_length = len(X_train[batch])
                except:
                    input_length = (X_train[batch]).shape[0]
                i = 0
                printProgressBar(0,
                                 input_length,
                                 prefix='Progress:',
                                 suffix='Complete',
                                 length=50)
            x_batch = X_train[batch].todense() if type(X_train[batch]) == csr_matrix else X_train[batch]
            for x in reversed(x_batch):

                grad, state, dy, dx = self.node.backward(
                    dy, csr_matrix(x), state, return_input=True)
                for k in grad.keys():
                    grads[k] = csr_matrix(grads[k] + grad[k])

                if verbose == 1:
                    i += 1
                    printProgressBar(i,
                                     input_length,
                                     prefix='Progress:',
                                     suffix='Complete', length=50
                                     )

            self.node.update_wieghts(grads, learning_rate, add_noise=add_noise)
            dInput.append(dx)

        return dInput

    def predict(self, x_in, state):
        '''
        predicts the output for a list of inputs
        '''
        for x in x_in:
            state = self.node.forward(csr_matrix(x), state, training=False)
        return state[0]


class Model:
    '''
    This class defines a model constructed with multiple layers 
    '''

    def __init__(self, units, num_features):
        '''
        Constructs the model 

        :param units: number of hidden units 
        :num_features: number of features in the input vector
        '''
        self.units = units
        self.units2 = int(units/2)

        self.lstm1 = LSTM_NN(units, num_features)
        self.lstm2 = LSTM_NN(self.units2, units)
        self.dense = Dense(self.units)

    def load_pickle(self, model):
        '''
        Initializes the wieghts of the model with 
        the wieghts of an old model stored in another object.
        '''
        self.units = model.units
        self.units2 = model.units2
        
        self.lstm1.load_pickle(model.lstm1)
        self.lstm2.load_pickle(model.lstm2)

        self.dense.load_pickle(model.dense)

    def load_pickle_from_file(self, file_name:str):
        '''
        Initializes the wieghts of the model with 
        the wieghts of an old model stored in a pickle file.
        '''
        with open(file_name, 'rb') as file:
            model = pickle.load(file)

        self.units = model.units
        self.units2 = model.units2

        self.lstm1.load_pickle(model.lstm1)
        self.lstm2.load_pickle(model.lstm2)

        self.dense.load_pickle(model.dense)

    def train(self, train_x, train_y, batch_size, learning_rate=0.1, add_noise=False):
        '''
        This method performs the trianing in the model. In this method both the forward and the backward propagations are preformed. 

        :param train_x: the input training data
        :param train_y: the labels of the training data
        :param batch_size: the size of the training batch 
        :param learning_rate: the learning rate used to update the weights 
        :param add_noise: determines if the model should add some noise to the gradients before updating the weights. 

        :returns: the final loss of the model 
        '''
        i = 0
        loss = 0
        start_state = [np.zeros((1, self.units)),
                       np.zeros((1, self.units))]

        start_state2 = [np.zeros((1, self.units2)),
                        np.zeros((1, self.units2))]

        data = list(zip(train_x, train_y))

        training_batch = [data[x:x+batch_size]
                          for x in range(0, len(data), batch_size)]
        epoch = 0

        total_epochs = math.ceil(len(data)/batch_size)

        for batch in training_batch:
            print()
            print("-----------------------------", " batch : ", epoch,
                  "/", total_epochs, "-----------------------------")
            print()
            epoch += 1

            i = 0

            self.cache = []
            preds = []
            inter_out = []

            x_batch, y_batch = list(zip(*batch))

            #if add_noise:
            #    for x in x_batch:
            #        x = x + sparse.random(x.shape[0], x.shape[1],density=0.1, format='csr')

            output = self.lstm1.forward_step(
                batch_size, x_batch, start_state, return_sequnece=False)
            #output2 = self.lstm2.forward_step(batch_size, output, start_state2)
            dense_output = self.dense.forward_step(output, batch_size)

            avg_loss = 0
            for pred, real in list(zip(dense_output, y_batch)):
                avg_loss += pred - real

            print(list(zip(dense_output, y_batch)))

            avg_loss /= len(batch)
            print("average Loss: ", avg_loss)

            print("adjusting the weights...")

            doutput = self.dense.backward_step(
                output, dense_output, y_batch, learning_rate)
            #doutput2 = self.lstm2.backward_step(
            #    batch_size, output, doutput, start_state2, learning_rate=learning_rate, add_noise=add_noise)
            self.lstm1.backward_step(
                batch_size, x_batch, doutput, start_state, learning_rate=learning_rate, add_noise=add_noise)

        return loss

    def predict(self, x):
        '''
        Predicts the output of the input x.
        '''
        state1 = [np.zeros((1, self.units)),
                  np.zeros((1, self.units))]

        pred = self.lstm1.predict(x, state1)
        pred = self.dense.forward_step(pred, 1)
        return pred

    def predict_list(self, X_values):
        '''
        Predicts the output of all the input values in the array X_values.
        '''
        preds = []
        for x in X_values:
            pred = self.predict(x)
            preds.append(pred)

        return preds