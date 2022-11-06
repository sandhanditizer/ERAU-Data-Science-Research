"""
Zachariah Kline 
Updated: 6 Nov 2022
"""

from sklearn.decomposition import NMF
from data import Data

# Rank size is determined by n_components.
# init is what algorithm we used for the approximation.
# max_iter controls how long you want the algorithm to spend on the approximation.
nnmf = NMF(n_components=20, init='nndsvd', max_iter=10000)
data = Data(split_type='rand', training_size=0.8)

W_train = nnmf.fit_transform(data.absorb_train)
H_train = nnmf.components_
W_test = nnmf.transform(data.absorb_test)
H_test = nnmf.components_