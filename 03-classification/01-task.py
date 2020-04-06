# %%
import dataset
import importlib
importlib.reload(dataset)
from dataset import *

# %%
X_train, X_test, y_train, y_test = get_data()

# %%
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_jobs=20)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_neighbors": [3, 4, 5], "weights": ["distance", "uniform"]}
]

grid_serach = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_serach.fit(X_train, y_train)

# %%
grid_serach.best_score_

# %%
grid_serach.best_params_
