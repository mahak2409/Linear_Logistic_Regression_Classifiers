import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
class Regression(object):
    """docstring for Regression."""
    def __init__(self):
        super(Regression, self).__init__()
        self.model = LinearRegression()


    """You can give any required inputs to the fit()"""
    def fit(self,X,y):
        self.model.fit(X,y)

        """Here you can use the fit() from the LinearRegression of sklearn"""


    """ You can add as many methods according to your requirements, but training must be using fit(), and testing must be with predict()"""
    

    def predict(self,X_test):
      x_numpy = X_test.to_numpy()
      y_predicted = np.dot(x_numpy,self.model.coef_.T)+self.model.intercept_
      """ Write it from scratch usig oitcomes of fit()"""

      """Fill your code here. predict() should only take X_test and return predictions."""


      return y_predicted