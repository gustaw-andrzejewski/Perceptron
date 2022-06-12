import numpy as np
import yaml
import csv
import matplotlib.pyplot as plt
import sys
from timeit import default_timer as timer
import random

class Perceptron:

    def __init__(self, epochs=10, learning_rate=0.1, activation='relu'):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation = activation
        random_seed = 42
        np.random.seed(random_seed)

    
    def activation_f(self, x):
        def sigmoid(x):
            sig = 1 / (1 + np.exp(-x))
            return sig
        if self.activation == 'relu':
            if x > 0:
                activation_f = x
            else:
                activation_f = 0
        elif self.activation == 'sigmoid':
            activation_f = sigmoid(x)
        elif self.activation == 'tanh':
            activation_f = np.tanh(x)
        else:
            sys.exit('Error: Unknown activation function')

        return activation_f

    
    def activation_f_d(self, x):
        def sigmoid_d(x):
            sig = np.exp(-x) / (1 + np.exp(-x))**2
            return sig
        if self.activation == 'relu':
            if x > 0:
                activation_f = 1
            else:
                activation_f = 0
        elif self.activation == 'sigmoid':
            activation_f = sigmoid_d(x)
        elif self.activation == 'tanh':
            activation_f = np.tanh(x)
        else:
            sys.exit('Error: Unknown activation function')

        return activation_f


    def read_data(self, filename, normalize=True):
        with open(filename, 'r') as file:
            data = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            X, Y = [],[]
            for line in data:
                X.append(line[:-1])
                Y.append(line[-1:])

        self.Nin = len(X[0])

        if normalize:
            X,Y = self.normalize(X,Y)
        
        return X,Y
    

    def normalize(self, X, Y):
        self.min_val = min(np.min(X), np.min(Y))

        self.max_val = max(np.max(X), np.max(Y))

        X = (X - self.min_val) / (self.max_val - self.min_val)
        Y = (Y - self.min_val) / (self.max_val - self.min_val)

        return X, Y

    
    def normalize_back(self, *X):
        X_out = []
        for X_single in X:
            X_out.append([self.min_val + i * (self.max_val - self.min_val)
                    for i in X_single])

        return X_out

    

    def train_validation_split(self, X, Y, split=0.2):
        if split > 0.9:
            print('Warning: Adjusting to max split value = 0.9')
            split = 0.9

        if split < 0.1:
            print('Warning: Adjusting to min split value = 0.1')
            split = 0.1
        
        validation_data_size = int(split*len(X))

        temp = list(zip(X, Y))
        random.shuffle(temp)
        X, Y = zip(*temp)
        
        X_train,X_valid,Y_train,Y_valid = [],[],[],[]

        X_train = X[validation_data_size:]
        X_valid = X[:validation_data_size]
        Y_train = Y[validation_data_size:]
        Y_valid = Y[:validation_data_size]

        return X_train,X_valid,Y_train,Y_valid


    def initialize_weights(self):
        self.weights = np.random.random(self.Nin)
        print(self.weights)


    def train(self, X_train, Y_train, X_valid, Y_valid):

        start_time = timer()

        self.initialize_weights()

        RMSE_train = []
        RMSE_valid = []

        for epoch in range(self.epochs):
            print(f'Epoch = {epoch+1}')

            sumRMSE_train = 0
            for i in range(len(X_train)):
                sumWeighted = 0
                for j in range(self.Nin):
                    sumWeighted += self.weights[j]*X_train[i][j]
                Y_out = self.activation_f(sumWeighted)

                for j in range(self.Nin):
                    self.weights[j] += self.learning_rate *self.activation_f_d(sumWeighted) * (Y_train[i]-Y_out)*X_train[i][j]
                
                sumRMSE_train += (Y_out-Y_train[i])**2
            
            RMSE_train.append(np.sqrt(sumRMSE_train / len(X_train)))
            print(f'RMSE for training set = {RMSE_train[epoch]}')

            if len(X_valid) > 0:
                sumRMSE_valid = 0
                for i in range(len(X_valid)):
                    sumWeighted = 0
                    for j in range(self.Nin):
                        sumWeighted += self.weights[j]*X_valid[i][j]
                    Y_out = self.activation_f(sumWeighted)
                    sumRMSE_valid += (Y_out-Y_valid[i])**2

                RMSE_valid.append(np.sqrt(sumRMSE_valid / len(X_valid)))
                print(f'RMSE for validation set = {RMSE_valid[epoch]}')
        self.plot_training(RMSE_train, RMSE_valid)

    def test(self, X_test):

        Y = []

        for i in range(len(X_test)):
            sumWeighted = 0
            for j in range(self.Nin):
                sumWeighted += self.weights[j] * X_test[i][j]
            Y.append(self.activation_f(sumWeighted))

        return Y


    def plot_training(self, RMSE_train, RMSE_valid):
        plt.plot(RMSE_train, label = 'RMSE for training set')
        plt.plot(RMSE_valid, label = 'RMSE for validation set')
        plt.legend()
        plt.title('Training results')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.show()


    def save_model(self, filename):
        data = {'nin':self.Nin,
                'epochs':self.epochs,
                'learning_rate':self.learning_rate,
                'activation':self.activation,
                'min_val':float(self.min_val),
                'max_val':float(self.max_val),
                'weights':self.weights.tolist()}

        with open(filename, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        print('Model has been saved to file', filename)


    def load_model(self, filename):
        try:
            with open(filename, 'r') as stream:
                data = yaml.load(stream, Loader=yaml.Loader)
        except FileNotFoundError:
            sys.exit('Error: Model file does not exists.')

        try:
            self.Nin = data['nin']
            self.epochs = data['epochs']
            self.learning_rate = data['learning_rate']
            self.activation = data['activation']
            self.min_val = np.float64(data['min_val'])
            self.max_val = np.float64(data['max_val'])
            self.weights = np.array(data['weights'])
        except KeyError:
            sys.exit('Error: Wrong format of the model file.')

