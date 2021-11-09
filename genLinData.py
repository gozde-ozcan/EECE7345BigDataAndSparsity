import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """ 
    """

    n = 2000 # number of samples
    d = 1000 # number of features
    sigma = 0.1 # noise scaler
    print('n = ', n, 'd = ', d, 'sigma = ', sigma)

    beta = np.random.rand(d)
    M = 1.0 / np.linalg.norm(beta) # beta scaler so that ||ß||^2 = 1
    beta = M*beta
    print('Generated beta has norms: l1=', np.linalg.norm(beta, 1), 'l2=', np.linalg.norm(beta))

    # Training data generation
    X_tr = np.random.randn(n, d)
    y_tr = X_tr.dot(beta) + np.random.randn(n)*sigma
    xpand_X_tr = []
    print(X_tr.shape)

    # Test data generation
    X_te = np.random.randn(n//9, d)  # 90% Train 10% test split
    y_te = X_te.dot(beta) + np.random.randn(n//9)*sigma

    beta = np.array(list(beta))
    print("Test error of true model is: ", np.mean((y_te - X_te.dot(beta))**2))

    MSE = []
    for N in range (1, 2000):
        data = X_tr[:N, :]
        y = y_tr[:N]

        estim_beta = np.matmul(np.linalg.pinv(data), y) # ß^ = X†y, where † denotes the Moore–Penrose pseudoinverse
        err = np.mean((y_te - X_te.dot(estim_beta))**2)
        MSE.append(err)

    plt.figure()
    plt.plot(range(1, 2000), MSE, marker="o")
    plt.axvline(x=1000, color='r', linestyle='--')
    plt.title("Test Risk vs. Samples")
    plt.xlabel("number of samples")
    plt.ylabel("MSE")
    plt.show()
    plt.savefig('results.png', bbox_inches='tight')
    plt.savefig('results.pdf', bbox_inches='tight')

