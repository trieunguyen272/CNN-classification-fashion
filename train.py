from keras.utils import np_utils
import numpy as np
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import fashion_mnist
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


class CNN:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (height, width, depth)
            chanDim = 1
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


NUM_EPOCHS = 10
INIT_LR = 1e-2
BATCH_SIZE = 32

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
else:
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


print("Copiling Model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model = CNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=['accuracy']),

print("Training model...")
history = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

print(model.summary())

loss_train, accuracy_train = model.evaluate(X_train, y_train)

print("Loss_Train: ", loss_train)
print("Accuracy_Train: ", accuracy_train)

loss_test, accuracy_test = model.evaluate(X_test, y_test)

print("Loss_Test: ", loss_test)
print("Accuracy_Test: ", accuracy_test)

plt.figure()
plt.plot(np.arange(0, NUM_EPOCHS), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, NUM_EPOCHS),
         history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, NUM_EPOCHS),
         history.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, NUM_EPOCHS),
         history.history['val_accuracy'], label='val_acc')
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc='upper left')
plt.tight_layout()

model.save('static/models/fashion_mnist_cnn_model.h5')
