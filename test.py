import time
import numpy as np
import matplotlib.pyplot as plt

from xmeans import XMeans

if __name__ == "__main__":
    tt = 0
    for _ in range(100):
        x = np.array([np.random.normal(loc, 0.1, 20) for loc in np.repeat([1,2], 2)]).flatten()
        y = np.array([np.random.normal(loc, 0.1, 20) for loc in np.tile([1,2], 2)]).flatten()

        st = time.time()
        x_means = XMeans(random_state = 1).fit(np.c_[x,y])
        tt += time.time() - st
        print(tt / 100)
        #print(x_means.labels_)
        #print(x_means.cluster_centers_)

    plt.scatter(x, y, c = x_means.labels_, label = "data", s = 30)
    plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "+", label = "center", s = 100)
    plt.xlim(0, 3)
    plt.ylim(0, 3)
    plt.legend()
    plt.grid()
    plt.show()
