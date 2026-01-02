from sklearn.linear_model import LogisticRegression

class Classifier:
    def __init__(self, train_x, train_y, test_x, test_y) -> None:
        self.__train_x = train_x
        self.__train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.__logical_regression_classifier = LogisticRegression(random_state=16)
        self.Fit(self.__logical_regression_classifier)
        prediction = self.Predict(self.__logical_regression_classifier)
        self.Plot_ConfusionMatrix(prediction, test_y)
        ######################## classifiers to be implemented ####################################
    def Fit(self, estimator):
        estimator.fit(self.__train_x, self.__train_y)
        ######################## classifiers to be implemented ####################################
    def Predict(self, estimator):
        ########################To be implemented####################################
        pass
    def Score(self, data, true_labels):
        ########################To be implemented####################################
        pass
    def Plot_ConfusionMatrix(self, predictions, true_labels):
        ########################To be implemented####################################
        pass
