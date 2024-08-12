import numpy as np
# from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def ranking(arr):
    ret_arr = np.copy(arr)
    rank_x = np.sort(arr)
    for i in range(len(rank_x)):
        for j in range(len(rank_x)):
            if rank_x[i] == arr[j]:
                ret_arr[j] = i
    return np.asarray(ret_arr)

def spr(x, y):
    x_rank = ranking(x)
    y_rank = ranking(y)
    mean_x = np.mean(x_rank)
    mean_y = np.mean(y_rank)
    # numerator = 0
    # for i in range(len(x)):
    #     numerator += (x_rank[i]-mean_x)*(y_rank[i]-mean_y)
    temp_x = (x_rank-mean_x)
    temp_y = (y_rank-mean_y)
    numerator = np.multiply(temp_x,temp_y)
    numerator = numerator.sum()
    den_x = 0
    den_y = 0
    for i in range(len(y)):
        den_x = den_x+(x_rank[i]-mean_x)*(x_rank[i]-mean_x)
        
        den_y = den_y+(y_rank[i]-mean_y)*(y_rank[i]-mean_y)
    denominator = np.sqrt(den_x*den_y)
    print(numerator)
    print(denominator)
    rho = numerator/denominator
    return rho


def main():
    mos = np.array([0.1, 0.3, 0.2, 0.5, 0.8, 1.0, 0.99, 0.45, 0.21, 0.74]) #x
    score = np.array([12, 32, 33, 62, 95, 89, 90, 50, 35, 70]) #y

    # plt.scatter(mos, score)
    # plt.show()

    print(spr(mos, score))
    # print(spearmanr(mos, score)[0])

def main2():
    num_samples = 500

    mos = np.random.rand(num_samples)
    score = np.clip(100 * np.power(mos, 4) + np.random.randn(num_samples) * 10, 0, 100)

    plt.scatter(mos, score, s=3)
    plt.show()

    print(spr(mos, score))
    # print(spearmanr(mos, score)[0])


if __name__ == '__main__':
    main()
    # main2()