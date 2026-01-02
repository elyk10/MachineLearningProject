from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

class Regressor:
    def __init__(self, train_x, train_y, test_x, test_y) -> None:
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.__logical_regression_classifier = LogisticRegression(random_state=16)
        self.Fit(self.__logical_regression_classifier)
        prediction = self.Predict(self.__logical_regression_classifier)
        logical_regression_score = self.Score(prediction, self.test_y, r2=True)
        ######################## Regressors to be implemented ####################################
    def Fit(self, estimator):
        estimator.fit(self.__train_x, self.__train_y)
        ######################## classifiers to be implemented ####################################
    def Predict(self, estimator):
        ########################To be implemented####################################
        pass
    def Score(self, data, true_labels, r2=False, mse=False, mae=False, rmse=False):
        if not r2:
            return r2_score(data, true_labels)
        ########################To be implemented####################################