import pandas as pd
from keras.models import Sequential
from keras.layers import *

# Loads the preprocessed data from the previous tutorial
training_data_df = pd.read_csv("sales_data_training_scaled.csv")

# Gets the input features from the preprocessed data and assigns it to variable X
X = training_data_df.drop('total_earnings', axis=1).values

# Gets the total_earnings column from the preprocessed data which we are going to make predictions on and assigns it to variable Y
Y = training_data_df[['total_earnings']].values

# ------------------------Define the model-------------------------------------------

# We create a sequential model by declaring a new Sequential object
model = Sequential()

# Creating the first layer of out neural network
# A dense layer means that every node in this layer in connected to every node in the previous layer
# input_dim is the number of input features we have in our input layer
# and we also define the activation function we are going to use process this layer
model.add(Dense(50,input_dim=9,activation="relu"))

# Creating the second layer
model.add(Dense(100,activation="relu"))

# Creating the third layer
model.add(Dense(50,activation="relu"))

# Output layer will have exacty one node in  it as we going to predict only one feature which is the amount of money each game would make
# sine our predicted value should be a single linear value so we use the linear_activation function
model.add(Dense(1,activation="linear"))


# Compile the model
# Loss function is how keras measures how close are the predicted output to the expected output
# Most common is mean_squared_error
# optimizer defines the optimization algorithm we are going to use , most common is adam optimizer
model.compile(loss="mean_squared_error", optimizer="adam")


