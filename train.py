from perceptron import Perceptron

if __name__ == '__main__':

    p = Perceptron()

    X,Y = p.read_data('training_data.csv')
    X_train, X_valid, Y_train, Y_valid = p.train_validation_split(X, Y)

    p.train(X_train, Y_train, X_valid, Y_valid)

    p.save_model('Perceptron_model.yaml')