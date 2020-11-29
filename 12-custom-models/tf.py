import tensorflow as tf

#%%
tf.constant([[1., 2., 3.], [4., 5., 6.]])

#%%
tf.constant(42)

#%%
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
t.shape

#%%
t.dtype

#%%
t[:, 1:]

#%%
t[..., 1, tf.newaxis]

#%%
t + 10

#%%
tf.square(t)

#%%
tf.transpose(t)

#%%
from tensorflow import keras
K = keras.backend
K.square(K.transpose(t)) + 10

#%%
import numpy as np

a = np.array([2., 4., 5.])
tf.constant(a)

#%%
t.numpy()

#%%
tf.square(a)

#%%
tf.constant(2.) + tf.constant(40)

#%%
tf.constant(2.) + tf.constant(40, dtype=tf.float64)

#%%
t2 = tf.constant(40, dtype=tf.float64)
tf.constant(2.0) + tf.cast(t2, tf.float32)

#%%
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v

#%%
v.assign(2 * v)

#%%
v[0, 1].assign(42)

#%%
v[:, 2].assign([0., 1.])

#%%
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])



