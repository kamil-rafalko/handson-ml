# %%
import dataset
import importlib
importlib.reload(dataset)
from dataset import *

# %%
X_train, X_test, y_train, y_test = get_data()
some_digit = X_train[1000]
some_digit

# %%
some_digit_image = some_digit.reshape(28, 28)
some_digit_image

# %%
import matplotlib.pyplot as plt

draw_image(some_digit_image)

# %%

right_shifted_image = shift_one_pixel_in_direction(some_digit_image, ShiftDirection.UP)
draw_image(right_shifted_image)


# %%
import numpy as np

# reshape_to_image(X_train[:10])

another_digit_image = np.apply_over_axes(
    lambda image: shift_one_pixel_in_direction(image, ShiftDirection.UP), 
    np.apply_along_axis(lambda digit: digit.reshape(28, 28), 1, X_train),
    [1,2]
)[1001]


another_digit_image = np.apply_along_axis(lambda digit: digit.reshape(28, 28), 1, X_train)
another_digit_image.shape

# draw_image(another_digit_image)

# another_digit_image_shifted = shift_one_pixel_in_direction(another_digit_image, ShiftDirection.UP)

# draw_image(another_digit_image_shifted)


# shift_one_pixel_in_direction_v(reshape_to_image(X_train), ShiftDirection.UP)

# for direction in iter(ShiftDirection):
    
#     shift_one_pixel_in_direction_v(reshape_to_image(X_train), direction)   

# %%
a = np.arange(12).reshape((4,3))
a

# %%
