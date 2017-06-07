import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from xmeans import XMeans

N = 10

def test_1(visualize = False):
    x = np.array([np.random.normal(loc, 0.1, 20) for loc in np.repeat([1,2], 2)]).flatten()
    y = np.array([np.random.normal(loc, 0.1, 20) for loc in np.tile([1,2], 2)]).flatten()
    st = time.time()
    x_means = XMeans(random_state = 1).fit(np.c_[x, y])
    et = time.time() - st

    if visualize:
        print(x_means.labels_)
        print(x_means.cluster_centers_)

        colors = ["g", "b", "c", "m", "y", "b", "w"]
        for label in range(x_means.labels_.max()+1):
            plt.scatter(x[x_means.labels_ == label], y[x_means.labels_ == label], c = colors[label], label = "sample", s = 30)
        plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "+", label = "center", s = 100)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.title("x-means_test")
        plt.legend()
        plt.grid()
        plt.show()

    return et

def test_2(visualize = False):
    X, y = make_blobs(n_samples=500,n_features=2,centers=5,cluster_std=0.8,center_box=(-10.0, 10.0),shuffle=True,random_state=1)
    x, y = X[:,0], X[:,1]
    X = np.c_[x, y]
    st = time.time()
    x_means = XMeans(random_state = 1).fit(np.c_[X])
    et = time.time() - st

    if visualize:
        print(x_means.labels_)
        print(x_means.cluster_centers_)

        plt.scatter(x, y, c='black', marker='o', s=50)
        plt.title("x-means_sample")
        plt.grid()
        plt.show()

        colors = ["g", "b", "c", "m", "y", "b", "w"]
        for label in range(x_means.labels_.max()+1):
            plt.scatter(x[x_means.labels_ == label], y[x_means.labels_ == label], c = colors[label], label = "sample", s = 30)
        plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "+", label = "center", s = 250)
        plt.title("x-means_test")
        plt.legend()
        plt.grid()
        plt.show()

    return et

def test_3(visualize = False):
    X, y = make_blobs(n_samples=500,n_features=2,centers=8,cluster_std=1.5,center_box=(-10.0, 10.0),shuffle=True,random_state=1)
    x, y = X[:,0], X[:,1]
    X = np.c_[x, y]
    st = time.time()
    x_means = XMeans(random_state = 1).fit(np.c_[X])
    et = time.time() - st

    if visualize:
        print(x_means.labels_)
        print(x_means.cluster_centers_)

        plt.scatter(x, y, c='black', marker='o', s=50)
        plt.title("x-means_sample")
        plt.grid()
        plt.show()

        colors = ["g", "b", "c", "m", "y", "b", "w"]
        for label in range(x_means.labels_.max()+1):
            plt.scatter(x[x_means.labels_ == label], y[x_means.labels_ == label], c = colors[label], label = "sample", s = 30)
        plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "+", label = "center", s = 250)
        plt.title("x-means_test")
        plt.legend()
        plt.grid()
        plt.show()

    return et

if __name__ == "__main__":
    tt = 0
    for _ in range(N):
        tt += test_3()
    print(tt / N)

    test_3(True)
