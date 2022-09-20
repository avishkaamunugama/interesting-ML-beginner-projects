import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import *


# To find the optimum number of nodes we can iterate through a list of parameters and each time log the result with a
# different name

def train_model(node_list):
    training_data_df = pd.read_csv("sales_data_training_scaled.csv")

    X = training_data_df.drop('total_earnings', axis=1).values
    Y = training_data_df[['total_earnings']].values

    # Define the model
    model = Sequential()

    for i in node_list:
        model.add(Dense(i, input_dim=9, activation='relu'))
        for j in node_list:
            model.add(Dense(j, activation='relu'))
            for k in node_list:
                RUN_NAME = "{}-{}-{}".format(i, j, k)

                model.add(Dense(k, activation='relu'))

                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_squared_error', optimizer='adam')

                # Create a TensorBoard logger
                logger = keras.callbacks.TensorBoard(
                    log_dir='logs/{}'.format(RUN_NAME),
                    histogram_freq=5,
                    write_graph=True
                )

                # Train the model
                model.fit(
                    X,
                    Y,
                    epochs=50,
                    shuffle=True,
                    verbose=2,
                    callbacks=[logger]
                )

                # Load the separate test data set
                test_data_df = pd.read_csv("sales_data_test_scaled.csv")

                X_test = test_data_df.drop('total_earnings', axis=1).values
                Y_test = test_data_df[['total_earnings']].values

                test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
                print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


train_model([50, 100])


# After iteration though all the node values , it was found ou that the model performs best at 50 - 100 -50 and 50 epochs
