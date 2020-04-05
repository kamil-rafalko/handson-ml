# %%
import dataset
import importlib
importlib.reload(dataset)
from dataset import *

# %%
X_train, X_test, y_train, y_test = get_data()
X_train.shape

# %%
import numpy as np

X_train_with_shifted = np.copy(X_train)
y_train_with_shifted = np.copy(y_train)
for direction in iter(ShiftDirection):
    X_train_with_shifted = np.append(X_train_with_shifted, np.apply_along_axis(lambda image: shift_one_pixel_in_direction(image, direction), 1, X_train), 0)
    y_train_with_shifted = np.append(y_train_with_shifted, y_train, 0)

X_train_with_shifted.shape

# %%
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=4, weights="distance", n_jobs=6)
knn_clf.fit(X_train_with_shifted, y_train_with_shifted)


# another_digit_image = another_digit.reshape(28, 28)

# draw_image(another_digit_image)

# another_digit_image_shifted = shift_one_pixel_in_direction(another_digit_image, ShiftDirection.UP)

# draw_image(another_digit_image_shifted)


# shift_one_pixel_in_direction_v(reshape_to_image(X_train), ShiftDirection.UP)

# for direction in iter(ShiftDirection):
    
#     shift_one_pixel_in_direction_v(reshape_to_image(X_train), direction)   



# %%
