from sklearn.datasets import fetch_openml
import numpy as np

def get_data():
    mnist = fetch_openml('mnist_784', version=1)
    print(mnist.keys())
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000],  y[60000:]
    return X_train, X_test, y_train, y_test

def draw_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()

from enum import Enum

class ShiftDirection(Enum):
    UP = [-1, 0]
    DOWN = [1, 0]
    LEFT = [0, -1]
    RIGHT = [0, 1]

from scipy.ndimage.interpolation import shift

def shift_one_pixel_in_direction(digit, direction):
    return shift(
        digit.reshape(28,28), 
        direction.value
    ).reshape(28 * 28)

