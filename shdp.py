import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial
from scipy.cluster.vq import kmeans2

class StickyHDPHMM:
    def __init__(self, data, alpha=1, kappa=1, gma=1,
                 nu=2, sigma_a=2, sigma_b=2, L=20, 
                 kmeans_init=False):
        """
        Fox, E. B., Sudderth, E. B., Jordan, M. I., 
        & Willsky, A. S. (2011). A sticky HDP-HMM 
        with application to speaker diarization. 
        The Annals of Applied Statistics, 1020-1056.
        
        X | Z = k ~ N(mu_k, sigma_k^2)
        mu_k ~ N(0, s^2)
        sigma_k ~ InvGamma(a, b)
        """
        
        self.L = L
        self.alpha = alpha
        self.gma = gma

        self.data = data
        self.kappa = kappa * data.size
        self.T, self.n = self.data.shape

        if kmeans_init:
            self.state = np.reshape(kmeans2(self.data.ravel(), L)[1], 
                                    self.data.shape)
        else:
            self.state = choice(self.L, self.data.shape)
            
        std = np.std(self.data)
        self.mu = normal(0, std, L)
        self.sigma = np.ones(L) * std
        for i in range(L):
            idx = np.where(self.state == i)
            if idx[0].size:
                cluster = self.data[idx]
                self.mu[i] = np.mean(cluster)
                self.sigma[i] = np.std(cluster)
                
        
        stickbreaking = self._gem(gma)
        self.beta = np.array([next(stickbreaking) for i in range(L)])
        self.N = np.zeros((L, L))
        for t in range(1, self.T):
            for i in range(self.n):
                self.N[self.state[t-1, i], self.state[t, i]] += 1
        self.M = np.zeros(self.N.shape)
        self.PI = (self.N.T / (np.sum(self.N, axis=1) + 1e-7)).T
        
        # Hyperparameters
        self.nu = nu

        self.a = sigma_a
        self.b = sigma_b
        

    def _logphi(self, x, mu, sigma):
        """
        Compute log-likelihood.
        """
        return -((x - mu) / sigma) ** 2 / 2 - np.log(sigma)
        
    def _gem(self, gma):
        """
        Generate the stick-breaking process with parameter gma.
        """
        prev = 1
        while True:
            beta_k = beta(1, gma) * prev
            prev -= beta_k
            yield beta_k
            
    def generator(self):
        """
        Simulate data from the sticky HDP-HMM.
        """
        self.state = [list(np.where(multinomial(1, dirichlet(self.beta), self.N))[1])]
        for i in range(1, self.T):
            self.state.append(list(np.where(multinomial(1, self.PI[i, :]))[0][0] for i in self.state[-1]))
            
        for i in range(self.T):
            self.data.append([normal(self.clusterPars[j][0], 
                              self.clusterPars[j][1]) for j in self.state[i]])

        self.state = np.array(self.state)
        self.data = np.array(self.data)
        
        
    def sampler(self):
        """
        Run blocked-Gibbs sampling
        """
        
        for obs in range(self.n):
            # Step 1: messages
            messages = np.zeros((self.T, self.L))
            messages[-1, :] = 1
            for t in range(self.T - 1, 0, -1):
                messages[t-1, :] = self.PI.dot(messages[t, :] * np.exp(self._logphi(self.data[t, obs], self.mu, self.sigma)))
                messages[t-1, :] /= np.max(messages[t-1, :])
            # Step 2: states by MH algorithm
            for t in range(1, self.T):
                j = choice(self.L) # proposal
                k = self.state[t, obs] 

                logprob_accept = (np.log(messages[t, k]) -
                                  np.log(messages[t, j]) +
                                  np.log(self.PI[self.state[t-1, obs], k]) -
                                  np.log(self.PI[self.state[t-1, obs], j]) +
                                  self._logphi(self.data[t-1, obs], 
                                               self.mu[k], 
                                               self.sigma[k]) -
                                  self._logphi(self.data[t-1, obs], 
                                               self.mu[j], 
                                               self.sigma[j]))
                if exponential(1) > logprob_accept:
                    self.state[t, obs] = j
                    self.N[self.state[t-1, obs], j] += 1
                    self.N[self.state[t-1, obs], k] -= 1            
        
        # Step 3: auxiliary variables
        P = np.tile(self.beta, (self.L, 1)) + self.n
        np.fill_diagonal(P, np.diag(P) + self.kappa)
        P = 1 - self.n / P
        for i in range(self.L):
            for j in range(self.L):
                self.M[i, j] = binomial(self.M[i, j], P[i, j])

        w = np.array([binomial(self.M[i, i], 1 / (1 + self.beta[i])) for i in range(self.L)])
        m_bar = np.sum(self.M, axis=0) - w
        
        # Step 4: beta and parameters of clusters
        self.beta = dirichlet(np.ones(self.L) * (self.gma / self.L + m_bar))

        # Step 5: transition matrix
        self.PI =  np.tile(self.alpha * self.beta, (self.L, 1)) + self.N
        np.fill_diagonal(self.PI, np.diag(self.PI) + self.kappa)
        for i in range(self.L):
            self.PI[i, :] = dirichlet(self.PI[i, :])
            idx = np.where(self.state == i)
            cluster = self.data[idx]
            nc = cluster.size
            if nc:
                xmean = np.mean(cluster)
                self.mu[i] = xmean / (self.nu / nc + 1)
                self.sigma[i] = (2 * self.b + (nc - 1) * np.var(cluster) + 
                                 nc * xmean ** 2 / (self.nu + nc)) / (2 * self.a + nc - 1)
            else:
                self.mu[i] = normal(0, np.sqrt(self.nu))
                self.sigma[i] = 1 / gamma(self.a, self.b)
                        
    def getPath(self, h):
        """
        Get the estimated sample path of h.
        """
        paths = np.zeros(self.data.shape[0])
        for i, mu in enumerate(self.mu):
            paths[np.where(self.state[:, h] == i)] = mu
        return paths