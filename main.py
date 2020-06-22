import cv2
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.datasets import mnist

from model import create_model


def train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)

    model = create_model()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)

    # Save model for prediction
    model.save('model.h5')


def predict():
    image = cv2.imread('mnistry-data.png', 0)
    model = keras.models.load_model('model.h5')
    file = open('result.txt', 'w')
    print(image.shape)
    for i in range(image.shape[0] // 28):
        for j in range(image.shape[1] // 28):
            cut = image[i*28:(i+1)*28, j*28:(j+1)*28]
            cut = np.expand_dims(cut, axis=3)
            cut = np.expand_dims(cut, axis=0)
            result = np.argmax(model.predict(np.array(cut)))
            file.write(result.__str__())
        print("%s/%s" % (i * j, (7476 * 5320) // (28 * 28)))
        file.write("\n")
    file.close()


if __name__ == '__main__':
    train()
    predict()
