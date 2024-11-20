from sklearn import linear_model as lm

class OLS():
    def __new__(self):
        return lm.LinearRegression(fit_intercept=True, copy_X=True)

class RidgeRegression():
    def __new__(self, seed, alpha = 0.01, max_iters = 500):
        return lm.Ridge(alpha=alpha, fit_intercept=True, copy_X=True, max_iter=max_iters, solver='lsqr', random_state=seed)