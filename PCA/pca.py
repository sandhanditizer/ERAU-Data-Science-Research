"""
Zachariah Kline
Updated: 6 Nov 2022
"""

from sklearn.decomposition import PCA
from data import Data


# n_components determines what dimension to project your data matrix into.
pca = PCA(n_components=20)
data = Data(split_type='rand', training_size=0.8)

transformed = pca.fit_transform(data.absorb_measurements_pos) # Not using the train/test split function.
transformed_train = pca.fit_transform(data.absorb_train)
transformed_test = pca.transform(data.absorb_test)