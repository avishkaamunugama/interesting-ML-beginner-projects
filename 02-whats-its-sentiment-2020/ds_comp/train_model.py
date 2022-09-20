import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from pathlib import Path
import matplotlib.pyplot as plt


train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))


print(train_examples[:10])


print(train_labels[:10])


model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples[:3])




model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_val = train_examples[:10000]
partial_x_train = train_examples[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


print(x_val.shape)

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)


history_dict = history.history
history_dict.keys()


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# saves the model structure and weights separately to reduce model size
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)
model.save_weights("model_weights.h5")

print("Saved model to disk.")






















#
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, BatchNormalization
# from keras.applications import vgg16
# from pathlib import Path
# import joblib
#
# # Loads the extracted features
# x_train = joblib.load("x_train.dat")
# y_train = joblib.load("y_train.dat")
# x_test = joblib.load("x_test.dat")
# y_test = joblib.load("y_test.dat")
#
# # Builds a sequential model
# model = Sequential()
# feature_extractor = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x_train = feature_extractor.predict(x_train)
# x_test = feature_extractor.predict(x_test)
# model.add(Flatten(input_shape=x_train.shape[1:]))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(6, activation='softmax'))
#
# # Compiles the model
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# # trains the model
# model.fit(
#     x_train,
#     y_train,
#     batch_size=64,
#     epochs=15,
#     validation_data=(x_test, y_test),
#     shuffle=True
# )
#
# # saves the model structure and weights separately to reduce model size
# model_structure = model.to_json()
# f = Path("model_structure.json")
# f.write_text(model_structure)
# model.save_weights("model_weights.h5")
#
# print("Saved model to disk.")
