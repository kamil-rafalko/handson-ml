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

knn_clf = KNeighborsClassifier(n_jobs=20)


# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_neighbors": [4], "weights": ["distance"]}
]

grid_serach = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_serach.fit(X_train_with_shifted, y_train_with_shifted)

# %%
grid_serach.best_score_

# %%
grid_serach.best_params_
