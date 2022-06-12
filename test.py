from perceptron import Perceptron
import numpy as np

if __name__ == '__main__':

    p = Perceptron()
    p.load_model('Perceptron_model.yaml')


    X_test, Y_test = p.read_data('test_data.csv')
    Y_out = p.test(X_test)

    X_test, Y_out, Y_test = p.normalize_back(X_test, Y_out, Y_test)

    print('Test results:')
    for i in range(len(Y_out)):
        print(f'{X_test[i][0]} + {X_test[i][1]} = {Y_out[i]} (expected {Y_test[i]})')
        sse = sum((np.array(Y_test) - np.array(Y_out))**2)
        tse = (len(Y_test) - 1) * np.var(Y_test, ddof=1)
        rmse = np.sqrt(sse / len(Y_out))
        r2_score = 1 - (sse / tse)
        print(f"\nRMSE score = {rmse}")
        print(f"R squared score = {r2_score}")