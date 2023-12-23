import numpy as np
import os
import time
from sys import stderr
import pickle
import matplotlib.pyplot as plt


class Dense:
    def __init__(self, neurons_num=0, activation='', input_shape=0, name='', seed=1):
        self.neurons_num = neurons_num  ## output number
        _activation_list = ['sigmoid', 'tanh', 'relu']
        self.weight_init = None
        self.scale = None
        self.optimization, self.momentum, self.adam = None, {"dW": None, "db": None}, {"dW": None, "db": None}
        self.momentum_corrected, self.adam_corrected = {"dW": None, "db": None}, {"dW": None, "db": None}
        self.beta1, self.beta2, self.epsilon, self.t = None, None, None, 0
        self.decay_rate = None
        self.delay_interval = None
        self.eta_change = []

        self.activation = activation
        if self.activation != '':
            if self.activation not in _activation_list:
                print('\n')
                stderr.write('Error in Dense %s: activation %s does not exist' % (name, activation))
                stderr.flush()
                exit()
        else:
            print('\n')
            stderr.write('Error in Dense %s: activation is not given' % name)
            stderr.flush()
            exit()

        if neurons_num == 0:
            print('\n')
            stderr.write('Error in Dense %s: pass neurons number' % self.name)
            stderr.flush()
            exit()

        self.input_shape = 0
        ## only for first layer
        if input_shape:
            self.input_shape = input_shape

        self.name = name
        self.random = np.random.RandomState(seed=seed)

        self.b, self.W, self.Z, self.A = None, None, None, None
        self.db, self.dW, self.dZ, self.dA = None, None, None, None

        self.eta, self.order_num = None, None

    def create_values(self, order_num: int, n_input: int = 0):
        if n_input:
            self.input_shape = n_input

        self.order_num = order_num

        self.b = np.zeros((self.neurons_num, 1))

        if self.weight_init == "random":
            self.W = self.random.randn(self.neurons_num, self.input_shape) * self.scale
        elif self.weight_init == "normal_distribution":
            self.W = self.random.normal(loc=0.0, scale=self.scale, size=(self.neurons_num, self.input_shape))
        elif self.weight_init == "he":
            self.W = self.random.randn(self.neurons_num, self.input_shape) * np.sqrt(np.divide(2., self.input_shape))
        elif self.weight_init == "xavier":
            self.W = self.random.randn(self.neurons_num, self.input_shape) * np.sqrt(np.divide(1., self.input_shape))


    def compile(self, eta: float, initialization: str = "random", scale: float = 0.1,
                eta_decay_rate: float = None, delay_interval: int = None):
        self.scale = scale
        _weight_init_list = ["random", "normal_distribution", "he", "xavier"]
        if initialization not in _weight_init_list:
            print('\n')
            stderr.write('Error in Dense %s: weight initialization "%s" does not exist. \n   Choose from: %s'
                         % (self.name, initialization, _weight_init_list))
            stderr.flush()
            exit()
        self.eta = eta
        self.eta_change.append(eta)
        self.weight_init = initialization

        if eta_decay_rate is not None and delay_interval is not None:
            self.decay_rate = eta_decay_rate
            self.delay_interval = delay_interval


    def optimization_init(self, optimization: str = None, beta1: float = None,
                          beta2: float = None, epsilon: float = 1e-10):

        if optimization == "momentum":
            self.momentum["dW"], self.momentum['db'] = np.zeros(self.W.shape), np.zeros(self.b.shape)
        elif optimization == "adam":
            self.momentum["dW"], self.momentum['db'] = np.zeros(self.W.shape), np.zeros(self.b.shape)
            self.adam["dW"], self.adam['db'] = np.zeros(self.W.shape), np.zeros(self.b.shape)

        _optimization_list = ["momentum", "adam"]

        if optimization is not None:
            if optimization not in _optimization_list:
                print('\n')
                stderr.write(
                    'Error in Dense L%s: Optimization %s does not exist \nChoose one from: "momentum", "adam"' %
                    (self.order_num, optimization))
                stderr.flush()
                exit()
            else:
                self.optimization = optimization

        if beta1 is not None:
            self.beta1 = beta1
        if beta2 is not None:
            self.beta2 = beta2

        self.epsilon = epsilon


    def eta_decay(self, epoch_num: int):
        if self.decay_rate is not None and self.delay_interval is not None:
            if epoch_num % self.delay_interval == 0:
                eta = 1 / (1 + self.decay_rate * np.floor(epoch_num / self.delay_interval)) * self.eta
                if eta != self.eta:
                    self.eta_change.append(eta)
                self.eta = eta


    def print_parameters(self):
        print("\nLayer L%s (%s) parameters: " % (self.order_num, self.name))
        ele_list = [['b', self.b], ['W', self.W], ['A', self.A],
                    ['Z', self.Z], ['db', self.db],
                    ['dW', self.dW], ['dA', self.dA], ['dZ', self.dZ]]
        for e in ele_list:
            if e[1] is not None:
                print(e[0] + ': ' + str(e[1].shape))
            else:
                print(e[0] + ': None')


    def _activation(self, Z: np.ndarray = None, forward: bool = True):
        if forward:
            if self.activation == "sigmoid":
                self.A = 1. / (1. + np.exp(-np.clip(Z, -250, 250)))
            elif self.activation == "tanh":
                self.A = np.tanh(Z)
            elif self.activation == "relu":
                self.A = np.where(Z < 0, 0, Z)

        else:
            g = None
            if self.activation == "sigmoid":
                g = np.multiply(self.A, (1 - self.A))
            elif self.activation == "tanh":
                g = 1 - np.power(self.A, 2)
            elif self.activation == "relu":
                g = np.where(self.Z >= 0, 1, 0)

            return g


    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        self.Z = np.dot(self.W, A) + self.b
        self._activation(Z=self.Z, forward=True)

        return self.A


    def backward(self, m: int, activation_left: np.ndarray, activation_derivative: np.ndarray,
                 output_activation_derivative: np.ndarray = None, input_data: np.ndarray = None):

        if output_activation_derivative is not None:
            dA = output_activation_derivative
        else:
            ## derivative that is passed from layer to the right
            dA = activation_derivative

        if input_data is not None:
            A_left = input_data
        else:
            A_left = activation_left

        ## dA is from this layer
        self.dZ = np.multiply(dA, self._activation(forward=False))

        self.db = np.divide(np.sum(self.dZ, axis=1, keepdims=True), m)

        ## A from layer to the left
        self.dW = np.divide(np.dot(self.dZ, A_left.T), m)

        if self.optimization == "momentum":
            self.momentum["dW"] = np.multiply(self.momentum["dW"], self.beta1) + np.multiply((1 - self.beta1), self.dW)
            self.momentum["db"] = np.multiply(self.momentum["db"], self.beta1) + np.multiply((1 - self.beta1), self.db)

        elif self.optimization == "adam":
            self.momentum["dW"] = np.multiply(self.momentum["dW"], self.beta1) + np.multiply((1 - self.beta1), self.dW)
            self.momentum["db"] = np.multiply(self.momentum["db"], self.beta1) + np.multiply((1 - self.beta1), self.db)

            self.momentum_corrected["dW"] = np.divide(self.momentum["dW"], (1 - np.power(self.beta1, self.t)))
            self.momentum_corrected["db"] = np.divide(self.momentum["db"], (1 - np.power(self.beta1, self.t)))

            self.adam["dW"] = np.multiply(self.adam["dW"],
                                          self.beta2) + np.multiply((1 - self.beta2), np.power(self.dW, 2))
            self.adam["db"] = np.multiply(self.adam["db"],
                                          self.beta2) + np.multiply((1 - self.beta2), np.power(self.db, 2))

            self.adam_corrected["dW"] = np.divide(self.adam["dW"], (1 - np.power(self.beta2, self.t)))
            self.adam_corrected["db"] = np.divide(self.adam["db"], (1 - np.power(self.beta2, self.t)))

        if self.order_num > 1:
            dA_left = np.dot(self.W.T, self.dZ)
            return dA_left

        return None


    def actualization(self):
        if self.optimization is None:
            self.W = self.W - np.multiply(self.eta, self.dW)
            self.b = self.b - np.multiply(self.eta, self.db)
        elif self.optimization == "momentum":
            self.W = self.W - np.multiply(self.eta, self.momentum["dW"])
            self.b = self.b - np.multiply(self.eta, self.momentum["db"])
        elif self.optimization == "adam":
            self.W = self.W - np.multiply(self.eta, np.divide(self.momentum_corrected["dW"],
                                                              (np.sqrt(self.adam_corrected["dW"]) + self.epsilon)))
            self.b = self.b - np.multiply(self.eta, np.divide(self.momentum_corrected["db"],
                                                              (np.sqrt(self.adam_corrected["db"]) + self.epsilon)))


class Model:
    def __init__(self, layers, seed: int = 2):
        self.layers = {}
        self.start_time = None
        self.timer_1 = None
        self.timer_iter = []
        self.mean_time = 0
        self.threshold = None
        self.acc_bin, self.loss_bin = '  ', '  '

        self.train_data, self.train_label = None, None
        self.validation_data, self.validation_label = None, None

        self.loss_function = None

        self.eval_ = {'accuracy': [],
                      'val_accuracy': [],
                      'loss': [],
                      'val_loss': [],
                      }

        self.random = np.random.RandomState(seed)
        self.epochs = None

        for n in range(1, len(layers) + 1):
            # Naming layers
            layer_name = 'L' + str(n)
            self.layers[layer_name] = layers[n - 1]

            # Creating values on every layer
            last_layer = 'L' + str(n - 1)
            if n == 1:
                self.layers[layer_name].create_values(order_num=n, n_input=0)
            else:
                self.layers[layer_name].create_values(order_num=n, n_input=self.layers[last_layer].neurons_num)


    def compile(self, loss_function: str, eta: float = None, optimization: str = None, momentum_beta: float = None,
                adam_beta: float = None, threshold: int = 0.5):

        self.threshold = threshold

        loss_list = ["cross_entropy", "mean_squared_error"]
        if loss_function not in loss_list:
            print('\n')
            stderr.write('Error in Model: Loss Function %s does not exist \nChoose one from: %s' %
                         (loss_function, loss_list))
            stderr.flush()
            exit()
        self.loss_function = loss_function

        # compiling when learning rate is the same in all layers
        if eta:
            for l in self.layers:
                self.layers[l].compile(eta)

        _optimization_list = ["momentum", "adam"]
        if optimization is not None:
            if optimization not in _optimization_list:
                print('\n')
                stderr.write('Error in Model: Optimization %s does not exist \nChoose one from: "momentum", "adam"' %
                             optimization)
                stderr.flush()
                exit()
            if momentum_beta is None:
                print('\n')
                stderr.write('Error in Model: pass Momentum Beta hyperparameter')
                stderr.flush()
                exit()
            elif momentum_beta > 1 or momentum_beta < 0:
                print('\n')
                stderr.write('Error in Model: pass correct Momentum Beta hyperparameter')
                stderr.flush()
                exit()
            if optimization == "adam":
                if adam_beta is None:
                    print('\n')
                    stderr.write('Error in Model: pass Adam Beta hyperparameter')
                    stderr.flush()
                    exit()
                elif adam_beta > 1 or adam_beta < 0:
                    print('\n')
                    stderr.write('Error in Model: pass correct Adam Beta hyperparameter')
                    stderr.flush()
                    exit()

            for l in self.layers:
                if optimization == "momentum":
                    self.layers[l].optimization_init(optimization=optimization, beta1=momentum_beta)
                if optimization == "adam":
                    self.layers[l].optimization_init(optimization=optimization, beta1=momentum_beta, beta2=adam_beta)


    def display(self, test_accuracy: float = None):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[1].plot(range(self.epochs), self.eval_['accuracy'], label='Training Accuracy', color='blue')
        ax[1].plot(range(self.epochs), self.eval_['val_accuracy'], label='Validation Accuracy',
                   color='red', linestyle=':')

        if test_accuracy:
            print('\nTest Accuracy (optional) %.3f \n\n' % test_accuracy)

            ax[1].plot(range(self.epochs), [test_accuracy] * self.epochs, label='Test Accuracy', linestyle='dashdot',
                       color='green')

        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend(loc='lower right')
        ax[1].set_title('MNIST Classification Accuracy')

        if self.loss_function == "cross_entropy":
            ax[0].plot(range(self.epochs), self.eval_['loss'], label='Cross-Entropy Loss', color='pink')
            ax[0].plot(range(self.epochs), self.eval_['val_loss'], label='Validation Cross-Entropy Loss', color='purple')
        elif self.loss_function == "mean_squared_error":
            ax[0].plot(range(self.epochs), self.eval_['loss'], label='Mean Squared Error', color='pink')
            ax[0].plot(range(self.epochs), self.eval_['val_loss'], label='Validation Mean Squared Error', color='purple')

        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend(loc='upper right')
        ax[0].set_title('Loss Function')

        plt.show()


    def _cross_entropy_cost(self, A: np.ndarray, Y: np.ndarray, eval_: str = None) -> float:
        m = A.shape[1]
        cost = - np.divide(np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1.00000001 - A))), m)

        if eval_ == 'train':
            self.eval_['loss'].append(round(float(cost), 3))
        elif eval_ == 'validation':
            self.eval_['val_loss'].append(round(float(cost), 3))

        return round(float(cost), 3)


    def _mean_squared_error(self, A: np.ndarray, Y: np.ndarray, eval_: str = None) -> float:
        m = A.shape[1]
        cost = np.divide(np.sum((Y - A) ** 2), m)

        if eval_ == 'train':
            self.eval_['loss'].append(cost)
        else:
            self.eval_['val_loss'].append(cost)

        return float(np.round(cost, 4))


    def _accuracy(self, A: np.ndarray, Y: np.ndarray, eval_: str = None) -> float:
        a = np.where(A >= self.threshold, 1, 0)
        acc = np.mean(np.mean(np.where(a == Y, 1, 0)))

        if eval_ == 'train':
            self.eval_["accuracy"].append(np.round(acc, 3))
        elif eval_ == 'validation':
            self.eval_["val_accuracy"].append(np.round(acc, 3))

        return float(np.round(acc, 3))


    def predict(self, input_data: np.ndarray, adam_t: bool = False) -> np.ndarray:
        A = input_data
        for layer in self.layers:
            A = self.layers[layer].forward(X=A)

            if adam_t:
                self.layers[layer].t += 1

        # returning last activation (Y_Hat)
        return A


    def test_predict(self, test_data: np.ndarray, test_label: np.ndarray) -> float:

        pred = self.predict(test_data.T)

        acc = self._accuracy(pred, test_label.T)
        return acc


    def save_model(self, name: str, path: str = ''):
        if path != '':
            if not os.path.exists(path):
                os.mkdir(path)
        with open(os.path.join(path, name), 'wb') as file:
            pickle.dump(self, file)
            time.sleep(0.4)
            stderr.write('\nModel successfully saved! \n\n')
            stderr.flush()
            time.sleep(0.4)
        file.close()


    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, visualise: int = 1, shuffle: bool = True,
            validation_data=None, validation_label=None):

        self.epochs = epochs

        self.validation_data = validation_data.T
        self.validation_label = validation_label.T
        self.train_data = x.T
        self.train_label = y.T

        self.start_time = time.perf_counter()

        epoch_strlen = len(str(epochs))
        for i in range(epochs):

            if visualise != 0:
                stderr.write(f"\n\nEpoch: {i+1}/{epochs}\n")
                stderr.flush()
                time.sleep(0.1)
                progress = 0

            indices = np.arange(self.train_data.shape[1])

            if shuffle:
                self.random.shuffle(indices)

            # Creating batch slices
            self.timer_1 = time.perf_counter()
            for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
                batch_idx = indices[start_idx:start_idx + batch_size]

                if visualise != 0:

                    progress += 1
                    steps = int(np.round(self.train_data.shape[1] / batch_size, 0))

                    b = 0.8
                    self.timer_iter.append((time.perf_counter() - self.timer_1))
                    self.mean_time = b * self.mean_time + (1 - b) * (time.perf_counter() - self.timer_1)
                    mean_time_corrected = self.mean_time / (1 - np.power(b, len(self.timer_iter)))
                    time_to_finish = np.round(mean_time_corrected * (steps - progress), 1)
                    t = "sec"
                    if time_to_finish > 60:
                        time_to_finish /= 60
                        t = "min"

                    if progress == 1:
                        time_to_finish = '  '
                        t = ''

                    stderr.write(f"\r[{progress}/{steps}] │{'─'*int(progress/steps*50)+'ᐶ'} {'•'*(50 - int(progress/steps*50))}│ ETA: {time_to_finish}{t}  Training Accuracy: {self.acc_bin}  Loss: {self.loss_bin} ")
                    stderr.flush()
                    self.timer_1 = time.perf_counter()


                if self.layers["L1"].optimization == "adam":
                    adam = True
                else:
                    adam = False

                # train model forward with batch-data slices  # adam_t = True will increase t for adam optimization
                A = self.predict(input_data=self.train_data[:, batch_idx], adam_t=adam)

                dA = None
                # Backpropagation
                for layer in reversed(self.layers):

                    # in the last layer we need to pass output activation derivative by hand
                    if self.layers[layer].order_num == len(self.layers):

                        if self.loss_function == "cross_entropy":
                            oad = - np.divide(self.train_label[:, batch_idx], A) + \
                                  np.divide((1 - self.train_label[:, batch_idx]), (1.00000000001 - A))
                        elif self.loss_function == "mean_squared_error":
                            oad = -2 * (self.train_label[:, batch_idx] - A)
                        else:
                            oad = None
                        input_data = None
                        A_left = self.layers["L" + str(self.layers[layer].order_num - 1)].A

                    # first layer must to have data as an activation derivative from layer to left
                    elif self.layers[layer].order_num == 1:
                        oad, A_left = None, None
                        input_data = self.train_data[:, batch_idx]

                    # middle layers
                    else:
                        oad, input_data = None, None
                        A_left = self.layers["L" + str(self.layers[layer].order_num - 1)].A

                    dA = self.layers[layer].backward(m=batch_size, activation_left=A_left, activation_derivative=dA,
                                                     output_activation_derivative=oad, input_data=input_data)

                # Actualization of weights and biases
                for layer in self.layers:
                    self.layers[layer].actualization()

                small_pred = self.predict(input_data=self.train_data[:, batch_idx])
                self.acc_bin = self._accuracy(small_pred, self.train_label[:, batch_idx])
                if self.loss_function == "cross_entropy":
                    self.loss_bin = self._cross_entropy_cost(A=small_pred, Y=self.train_label[:, batch_idx])
                elif self.loss_function == "mean_squared_error":
                    self.loss_bin = self._mean_squared_error(A=small_pred, Y=self.train_label[:, batch_idx])



            for layer in self.layers:
                self.layers[layer].eta_decay(epoch_num=i + 1)

            train_pred = self.predict(input_data=self.train_data)
            self._accuracy(train_pred, self.train_label, eval_='train')

            if self.loss_function == "cross_entropy":
                self._cross_entropy_cost(A=train_pred, Y=self.train_label, eval_='train')
            elif self.loss_function == "mean_squared_error":
                self._mean_squared_error(A=train_pred, Y=self.train_label, eval_='train')

            valid_pred = self.predict(input_data=self.validation_data)
            self._accuracy(valid_pred, self.validation_label, eval_='validation')

            if self.loss_function == "cross_entropy":
                self._cross_entropy_cost(A=valid_pred, Y=self.validation_label, eval_='validation')
            elif self.loss_function == "mean_squared_error":
                self._mean_squared_error(A=valid_pred, Y=self.validation_label, eval_='validation')


            if visualise != 0:
                time.sleep(0.1)
                stderr.write(f"\r[{progress}/{steps}] │{'─' * int(progress / steps * 50) + 'ᐶ'}{'•' * (50 - int(progress / steps * 50))}│ ETA: {time_to_finish}{t}  Training Accuracy: {self.eval_['accuracy'][-1]}  Loss: {self.eval_['loss'][-1]}")
                time.sleep(0.1)
                stderr.write(f"\n   Validation Accuracy: {self.eval_['val_accuracy'][-1]}  Validation Loss: {self.eval_['val_loss'][-1]}")
                stderr.flush()
                time.sleep(0.1)


        print()
        for l in self.layers:
            print('Learning rate Notebook (Layer: %s): ' % l, self.layers[l].eta_change)

        print('\nLearning time: %.1f%s' % ((time.perf_counter() - self.start_time) / 60 if time.perf_counter() - self.start_time > 60 else time.perf_counter() - self.start_time,
                                           "min" if time.perf_counter() - self.start_time > 60 else 'sec'))
        return self
