"""
Zachariah Kline
Updated: 6 Nov 2022
"""

import matplotlib.pyplot as plt
from seaborn import heatmap
from nnmf import H_train, H_test


# The only good way to visuallize NNMF is to create a heat map. When creating
# a heat map, you can see that none of the vectors are sorted, unlike eigenvectors
# in PCA.
heatmap(H_train)
plt.show()

heatmap(H_test)
plt.show()
