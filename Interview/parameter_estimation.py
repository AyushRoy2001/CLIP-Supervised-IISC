import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_estimate(x):
    mean_req = np.mean(x, axis=1)
    return mean_req


def covariance_matrix_estimate(x):
    multi_ans = np.matmul((x.T-mean_estimate(x).T).T,((x.T-mean_estimate(x).T).T).T)
    
    return (multi_ans/999)


def estimate_params():
    mu = np.array([2.2, 9.8, 5.5])
    sigma = np.array([[2, 0.3, 0.95], [0.3, 1, -0.2], [0.95, -0.2, 2]])
    x = np.random.multivariate_normal(mu, sigma, 1000).T
    
    # plt.plot(x[:, 0])
    # plt.plot(x[:, 1])
    # plt.plot(x[:, 2])
    # plt.show()

    print(mean_estimate(x))
    print(covariance_matrix_estimate(x))


if __name__ == '__main__':
    estimate_params()
