from numpy import genfromtxt
import numpy as np
import statsmodels.api as sm
import math

class UnivariateForecasting:
    def __init__(self, train, steps, start):
        self.start = start
        self.steps = steps
        self.predictions = []
        self.errors = []
        self.rmse = []

        #set this row as the time series to be predicted
        self.train = train

        self.predictions = np.zeros(self.steps)
        self.errors = np.zeros(self.steps)
        self.rmse = 0
        self.p = 1
        self.q = 0


    def compute_rmse(self):
        self.rmse = math.sqrt(sum(err**2 for err in self.errors) / len(self.errors))

        print self.rmse
        return self.rmse

    def compute_arma(self):

        # try to predict for n steps
        rel = np.zeros(len(self.errors))
        for i in range(self.steps):
            print "step: " + str(i)
            train_data = self.train[0:self.start+i]

            res = sm.tsa.ARMA(train_data, order=(self.p, self.q)).fit()

            forecast = res.forecast(1)
            self.predictions[i] = forecast[0]

            self.errors[i] = forecast[0] - self.train[i+self.start]
            rel[i] = (forecast[0] - self.train[i+self.start]) / self.train[i+self.start]

        print self.predictions
        print "done - now compute for rmse"
        return self.predictions

    def compute_arma_all(self):

        # try to predict for n steps
        err = np.zeros(len(self.data))
        rel = np.zeros(len(self.data))
        for i in range(len(self.data)):
            self.train = self.data[i]
            res =  self.compute_arma()
            err[i] = res[0]
            rel[i] = res[1][0]

        print "rmse = "
        print self.calculateRootMeanSquareError(err)
        print "mae = "
        print self.calculateMeanAbsoluteError(err)
        print "rmae = "
        print self.calculateMeanAbsoluteError(rel)

    def calculateRootMeanSquareError(self, nums):
        ms = 0;
        for i in range(len(nums)):
            ms += nums[i] * nums[i]
        ms /= len(nums);
        return math.sqrt(ms);


    def calculateMeanAbsoluteError(self, nums):
        ms = 0;
        for i in range(len(nums)):
            ms += math.fabs(nums[i])
        return ms / len(nums)
