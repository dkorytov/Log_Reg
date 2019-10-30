

import numpy as np



class LogReg:
    def __init__(self):
        pass

    def fit(self, X, y, learning_rate=0.1):
        ''' blah, blah'''
        n, d = X.shape
        assert n == y.shape, 'X and y have different number of points'
        beta = np.ones(d)
        e = learning_rate
        for i in range(100):
            beta_grad = grad_loss_func(beta, X, y)
            beta = beta - e*beta_grad

        self.beta = beta

    def grad_loss_func(self, beta, X, y):
        n, dim = X.shape
        beta_grad = np.zeros(dim)
        denom_1 = np.dot(X,beta)
        denom_1 = 1-np.dot(X,beta)
        for d in dim:
            beta_grad[d] = np.sum(y*X[:,d]/denom_1 - (1-y)*X[:,d]/denom_2)
        return beta_grad
        
    def predict(self, X, threshold=0.5):
        prob = self.predict_prob(X)
        prediction = np.zeroes_like(prob, dtype=int)
        prediction[prob>threshold]=1
        return predict

    def predict_prob(self, X, beta=None):
        if beta is None:
            beta = self.beta    
        score = X*beta
        return self.sigmoid(x)
        
    def sigmoid(self, x):
        return np.exp(x)/(1.0 + np.exp(x))
        
    def loss_func(self, beta, X, y):
        y_hat = self.predict(X, beta=beta)
        loss = y*np.log(y_hat) + (1-y)*np.log(1-y_hat)
        return loss

    def grad_loss_func_brute(self, beta, X, y, ep=0.01):
        beta_grad = np.zeros(dim)
        loss0 = self.loss_func(beta, X, y)
        for d in dim:
            beta1 = beta.copy()
            beta1[d] +=ep
            loss1 = self.loss_func(beta1, X, y)
            beta_grad[d] = (loss1-loss0)/ep
        return beta_grad
