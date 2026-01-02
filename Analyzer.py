import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle


class Analyzer:
    def __init__(self) -> None:
        self.__data = None

    def read_dataset(self, fileName):
        self.__data  = pd.read_csv(fileName)

    def describe(self):
        print('numeric data description:')
        print(self.__data.describe())
        print('non numeric data description:')
        print(self.__data.describe(include='object'))

    def drop_missing_data(self):
        self.__data = self.__data.dropna()

    def plot_histograms_numeric(self, columnName):
        self.__data.hist(column=columnName, bins=50)
        plt.show()

    def retrieve_data(self):
        return self.__data
    
    def drop_columns(self, attribute_list):
        self.__data = self.__data.drop(attribute_list)
        
    def encode_features(self, columns_list):
        OHencoder = OneHotEncoder()
        self.__data[columns_list] = OHencoder.fit_transform(self.__data[columns_list])

    def encode_label(self, label):
        Lencoder = LabelEncoder()
        self.__data[label] = Lencoder.fit_transform(self.__data[label])

    def shuffle(self):
        self.__data = shuffle(self.__data).reset_index(drop = True)

    def sample(self, reduction_factor):
        if reduction_factor < 0.0 or reduction_factor > 1.0:
            raise ValueError("reduction_factor is outside limits of 0.0 and 1.0")
        
        self.__data = self.__data.sample(frac = reduction_factor)

    def plot_correlationMatrix(self):
        df = self.__data.corr(numeric_only = True)
        sns.heatmap(df, annot = True, cmap = "coolwarm", square = True)
        plt.title("Correlation Matrix")
        plt.show()

    def plot_pairPlot(self):
        sns.pairplot(self.__data)
        plt.title("Pair Plot")
        plt.show()

    def plot_histograms_categorical(self):
        columns = self.__data.select_dtypes(include = "object").columns

        for col in columns:
            counts =  self.__data[col].value_counts()

            plt.bar(counts.index.astype(str), counts.values)
            plt.title(f"Histogram of {col}")
            plt.xticks(rotation = 45)
            plt.show()

    def plot_boxPlot(self):
        columns = self.__data.select_dtypes(include = np.number).columns

        for col in columns:
            plt.boxplot(self.__data[col].dropna(), vert = True)
            plt.title(f"Box Plot of {col}")
            plt.show()