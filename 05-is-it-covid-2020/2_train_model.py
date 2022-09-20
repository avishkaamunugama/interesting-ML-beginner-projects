from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, PReLU
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

# Loads the extracted features
x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

# split data to train and test
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0, test_size=0.2)

# Builds a sequential model
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

# Compiles the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

# trains the model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=15,
    validation_data=(x_test, y_test),
    shuffle=True
)

# saves the model structure and weights separately to reduce model size
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)
model.save_weights("model_weights.h5")
print("Saved model to disk.")
