import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

class XMeans(KMeans):

    def __init__(self, k_init = 2, **k_means_args):
        self.k_init = k_init
        self.k_means_args = k_means_args

    def fit(self, X):
        km = KMeans(self.k_init, **self.k_means_args).fit(X)
        self.cluster_centers_ = list(km.cluster_centers_)
        self.labels_, _ = self.__recrucive_clustering(X, km.labels_, np.unique(km.labels_), np.max(km.labels_))
        self.cluster_centers_ = np.array(self.cluster_centers_)
        return self

    def __recrucive_clustering(self, X, labels, labelset, splits):
        for label in labelset:
            cluster = X[labels == label]
            if len(cluster) <= 3: continue
            km = KMeans(2, **self.k_means_args).fit(cluster)
            if self.__recrucive_decision(cluster, km.labels_, km.cluster_centers_):
                self.cluster_centers_[label] = km.cluster_centers_[0]
                self.cluster_centers_.append(km.cluster_centers_[1])
                km.labels_[km.labels_ == 1] += splits
                km.labels_[km.labels_ == 0] += label
                labels[labels == label], splits = self.__recrucive_clustering(cluster, km.labels_, [label, splits], splits+1)
        return labels, splits

    def __recrucive_decision(self, cluster, labels, centers):
        samples = cluster.shape[0]
        parameters = cluster.shape[1] * (cluster.shape[1] + 3) / 2
        bic = self.__bic(cluster, None, samples, parameters)
        new_bic = self.__bic(cluster[labels == 0], cluster[labels == 1], samples, parameters*2)
        return bic > new_bic

    def __bic(self, cluster_0, cluster_1, samples, parameters):
        if cluster_1 is not None:
            alpha = self.__model_likelihood(cluster_0, cluster_1)
            return -2.0 * (samples * np.log(alpha) + self.__log_likelihood(cluster_0) + self.__log_likelihood(cluster_1)) + parameters * np.log(samples)
        else:
            return -2.0 * self.__log_likelihood(cluster_0) + parameters * np.log(samples)

    def __model_likelihood(self, cluster_0, cluster_1):
        mu_0, mu_1 = np.mean(cluster_0, axis=0), np.mean(cluster_1, axis=0)
        sigma_0, sigma_1 = np.cov(cluster_0.T) , np.cov(cluster_1.T)
        beta = np.linalg.norm(mu_0 - mu_1) / np.sqrt(np.linalg.det(sigma_0) + np.linalg.det(sigma_1))
        return 0.5 / stats.norm.cdf(beta)

    def __log_likelihood(self, cluster):
        mu = np.mean(cluster, axis=0)
        sigma = np.cov(cluster.T)
        try:
            log_likehood = np.sum(stats.multivariate_normal.logpdf(x, mu, sigma) for x in cluster)
        except np.linalg.LinAlgError as err:
            sigma = sigma * np.identity(sigma.shape[0])
            log_likehood = np.sum(stats.multivariate_normal.logpdf(x, mu, sigma) for x in cluster)
        except ValueError as err:
            log_likehood = np.log(1.0)
        return log_likehood

    def predict(self):
        return self.labels_

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
