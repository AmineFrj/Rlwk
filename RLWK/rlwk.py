import numpy as np
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from coclust.clustering.spherical_kmeans import SphericalKmeans
import random
from collections import Counter

class Rlwk:
    def __init__(self, k, lambda_val=0.0001, alpha=0, beta=4, tmax=30, chi2=True, n_init=20, scale=False, init='r', z=None):
        self.k = k
        self.lambda_val = lambda_val
        self.alpha = alpha 
        self.beta = beta
        self.tmax = tmax
        self.chi2 = chi2
        self.n_init = n_init
        self.scale = scale
        self.init = init
        self.z = z

    def fit(self, X, verbose=False):
        n, p = X.shape
        weight = np.ones(p) / p
        w_dummy = np.zeros((p, self.k))

        # Initial labels
        if self.z is None :
            if self.init == 's':
                sk = SphericalKmeans(n_clusters=self.k, n_init=self.n_init)
                sk.fit(X)
                labels = sk.row_labels_
                labels = np.array(labels)
            elif self.init == 'r':
                labels = [random.randint(0, self.k-1) for i in range(n)]
                labels = np.array(labels)
            elif self.init == 'k':
                labels = KMeans(n_clusters=self.k, n_init=self.n_init).fit(X).labels_
            labels = labels.reshape(-1, 1)
        else : 
            labels = self.z.reshape(-1, 1)

        dist = np.zeros(self.k)
        D = np.zeros(p)
        for j in range(p):
            D[j] = 0

        on = np.ones(p)
        lambda_val = self.lambda_val / p**2
        
        ### Chi-squared
        if self.chi2:
            row_sums = X.sum(axis=1)
            col_sums_sqrt = np.sqrt(X.sum(axis=0))
            scaling_factors = np.outer(row_sums, col_sums_sqrt)
            X = X / scaling_factors
        ###      
                  
        # Scale
        if self.scale:
            X = preprocessing.scale(X, with_mean=False)
        
        # Calculate cluster centers
        M = np.zeros((self.k, p))
        for i in range(self.k):
            I = np.where(labels == i)[0]
            M[i, :] = np.mean(X[I, :], axis=0)
                
        # Calculate initial within distances
        for i in range(self.k):
            I = np.where(labels == i)[0]
            for j in I:
                D += self._vec_euc_dist(X[j, :], M[i, :], on)
                
        if self.alpha != 0:
            alpha = self.alpha
        else:
            D = 1.0 / D
            D = D**(1.0 / (self.beta - 1))
            alpha = np.sum(D)
            alpha = 1.0 / alpha
            alpha = alpha**(self.beta - 1)
        
        for iter in range(self.tmax):
            for i in range(self.k):
                I = np.where(labels == i)[0]
                # Empty class (choose point with furthest WSS)
                if len(I) == 0:
                    farthest_point = np.argmax(distances)
                    labels[farthest_point] = i
                    I = [farthest_point]
                    # In case there are multiple empty classes
                    distances[farthest_point] = -1
     
                M[i, :] = np.mean(X[I, :], axis=0)     
                    
            for j in range(p):
                D[j] = 0
            
            for i in range(self.k):
                I = np.where(labels == i)[0]
                for j in I:
                    D += self._vec_euc_dist(X[j, :], M[i, :], on)

            for i in range(p):
                if alpha > (lambda_val * D[i]):
                    weight[i] = alpha  / (D[i] + lambda_val)
                    weight[i] = weight[i]**(1.0 / (self.beta - 1))
                else:
                    weight[i] = 0
                        
            w_dummy = weight**self.beta + lambda_val * weight

            P2 = 0
            distances = []
            for i in range(n):
                for j in range(self.k):
                    dist[j] = self._wt_euc_dist(X[i, :], M[j, :], w_dummy)        
                labels[i] = np.argmin(dist)
                distances.append(dist[labels[i]])
                P2 += dist[labels[i][0]]
            P2 = P2/n - alpha * np.sum(weight)
            if verbose:
                print(P2)
            if iter == 0:
                P1 = P2
            elif P1 == P2:
                break
            else:
                P1 = P2
        
        L = np.sum(weight > 0)
        
        self.labels_ = [i[0] for i in labels]
        self.weights_ = weight
        self.L = L
        self.alpha = alpha
        self.P2 = P2
        #if len(labels)>0:
        #    self.nmi_score_ = normalized_mutual_info_score(self.labels_,y)
        #    self.ari_score_ = adjusted_rand_score(self.labels_,y)

    def get_top_features(self, feature_names, n_features=10):
        weigh_word = np.sum(self.weights_, axis=1)
        weigh_word = sorted(zip(feature_names, weigh_word), key=lambda x: x[1], reverse=True)
        
        return [i[0] for i in weigh_word[:n_features]]
    
    def _errors_calculate(self, T):
        n = np.sum(np.sum(T))
        k, k0 = T.shape
        flags = list(permutations(range(k)))
        match = np.zeros(len(flags))
        
        for j in range(len(flags)):
            Tnew = T[list(flags[j]), :]
            Tnew = Tnew[:min(k, k0), :min(k, k0)]
            match[j] = np.sum(np.diag(Tnew))
        
        maxmatch = np.max(match)
        Error = 1 - maxmatch / n
        flagrecord = flags[np.argmax(match)]
        
        return Error, flagrecord

    def _vec_euc_dist(self, x1, x2, w):
        p = (x1 - x2)**2
        p = w * p
        return p

    def _wt_euc_dist(self, x1, x2, w):
        p = (x1 - x2)**2
        p = np.sum(w * p)
        return p
