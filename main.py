from cryptocmd import CmcScraper
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression ,ARDRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import datetime


class PredictApp :
    def __init__(self):
        """
       this App predict future of Digital currency for 10 days
        """
        self.name = ""
        self.scaler = StandardScaler()
        self.linear_regression = LinearRegression()
        self.mlp_regressor = MLPRegressor(hidden_layer_sizes=60 ,max_iter=500)
        self.x_train = []
        self.y_train = []
        self.scaled_data = []
        self.today_date = datetime.datetime.now().day
        self.dates_list = []

    def find_currency(self, name: str = ""):
        """
        search for currency and show currency history
        :param name: currency name (* Use coin symbole words like BTC *)
        :return: dataframe of currency history with close price
        """
        self.name = name
        scraper = CmcScraper(self.name)
        dataframe = scraper.get_dataframe(date_as_index=True)
        dataframe = dataframe[["Close"]]
        dataframe.sort_index(inplace=True)
        print(dataframe)
        return dataframe

    def show_plot(self ,dataframe :pd.DataFrame):
        """
        Show currency plot
        :param dataframe: selected currency dataframe
        :return: None
        """
        plt.plot(dataframe)
        plt.title(self.name)
        plt.show()

    def predict_currency(self ,dataframe :pd.DataFrame):
        """
        Predict 10 days after today with linear regression and support vector machine and show plot of this days
        :param dataframe: selected currency dataframe
        :return: None
        """
        self.scaled_data = self.scaler.fit_transform(dataframe.values)
        split_size = len(self.scaled_data) * 90 // 100
        train_data = self.scaled_data[:, :]
        for i in range(60, len(train_data)):
            self.x_train.append(train_data[i - 60: i, :])
            self.y_train.append(train_data[i, 0])
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.y_train = self.y_train.reshape(-1)
        self.x_train = self.x_train.reshape(-1, 60)
        

        # del pred_data ,data
        self.linear_regression.fit(self.x_train, self.y_train)
        pred_lr = self.linear_regression.predict(self.x_train)
        print("lr score : ", mean_absolute_error(y_true=self.y_train, y_pred=pred_lr))
        pred_lr = self.scaler.inverse_transform(pred_lr)
        pred_array_lr = []
        size = len(self.scaled_data)
        for i in range(60):
            self.dates_list.append(self.today_date + i)
            data = self.scaled_data[size - (60 - i):, :]
            data = list(data.reshape(-1))
            data = np.array((data + pred_array_lr), dtype="float")
            data = data.reshape(1, 60)
            pred_data = self.linear_regression.predict(data)
            pred_array_lr.append(pred_data[0])
        pred_array_lr = self.scaler.inverse_transform(pred_array_lr)

        del pred_data ,data
        self.mlp_regressor.fit(self.x_train ,self.y_train)
        pred_nn = self.mlp_regressor.predict(self.x_train)
        print("nn score : ", mean_absolute_error(y_true=self.y_train, y_pred=pred_nn))
        pred_nn = self.scaler.inverse_transform(pred_nn)
        pred_array_nn = []
        size = len(self.scaled_data)
        for i in range(60):
            data = self.scaled_data[size - (60 - i):, :]
            data = list(data.reshape(-1))
            data = np.array((data + pred_array_nn), dtype="float")
            data = data.reshape(1, 60)
            pred_data = self.mlp_regressor.predict(data)
            pred_array_nn.append(pred_data[0])
        pred_array_nn = self.scaler.inverse_transform(pred_array_nn)


        plt.clf()
        plt.figure(self.name)

        plt.subplot(3 ,1 ,1)
        plt.title("Predict MLP")
        plt.plot(self.dates_list ,pred_array_nn)


        plt.subplot(3, 1, 2)
        plt.title("Predict Regression")
        plt.plot(self.dates_list ,pred_array_lr)

        plt.show()

def run_app():

    while True:
        app = PredictApp()
        name = input("Enter coin name (for exit type exit) : ")
        if name == "exit":
            break
        df = app.find_currency(name=name)
        while True:
            print("Commands : [plot , predict , back]")
            command = input("Enter Command : ")
            if command == "plot":
                app.show_plot(dataframe=df)
            if command == "predict":
                app.predict_currency(dataframe=df)
            if command == "back":
                break

    return None





if __name__ == "__main__" :

    # try :
    #     run_app()
    # except :
    #     print("can't find stock. try again !")
    #     run_app()
    run_app()
    
    
