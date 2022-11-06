"""
Zachariah Kline 
Updated: 6 Nov 2022
"""

from data import Data
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


data = Data(split_type='rand', training_size=0.8)
nnmf = NMF(n_components=20, init='nndsvd', max_iter=10000)
nnmf.fit_transform(data.absorb_measurements_pos)

# Plots the first 3 vectors in the coefficients matrix (H) against the wavelenghts.
plt.plot(data.wavelengths[0], nnmf.components_[2], color='goldenrod', linestyle='-.', label='Component 3')
plt.plot(data.wavelengths[0], nnmf.components_[1], color='dodgerblue', linestyle=':', label='Component 2')
plt.plot(data.wavelengths[0], nnmf.components_[0], color='darkgreen', label='Component 1')
plt.title('H Matrix Loadings vs. Wavelengths | NNMF')
plt.xlabel('Wavelengths (nm)')
plt.ylabel('Loadings')
plt.legend()
plt.show()

# Plotting the first 20 vectors in the H matrix against the selected wavelengths.
# Vectors 1-7
fig, ax = plt.subplots(7, figsize=(9,8))
for i in range(0, 7):
    ax[i].plot(data.wavelengths[0][80:1200], nnmf.components_[i][80:1200], color='blue')
    ax[i].spines['left'].set_position(('axes', 0.04))
    ax[i].spines['bottom'].set_position(('axes', -0.05))
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)

fig.tight_layout()
for i in range(0, 7):
    ax[i].set(ylabel=f'{i+1}')
    
fig.suptitle('H Matrix Loadings vs. Wavelengths | NNMF')
plt.show()

# ----------
# Vectors 8-14

fig, ax = plt.subplots(7, figsize=(9,8))
for i in range(7, 14):
    ax[i - 7].plot(data.wavelengths[0][80:1200], nnmf.components_[i][80:1200], color='blue')
    ax[i - 7].spines['left'].set_position(('axes', 0.04))
    ax[i - 7].spines['bottom'].set_position(('axes', -0.05))
    ax[i - 7].spines['top'].set_visible(False)
    ax[i - 7].spines['right'].set_visible(False)

fig.tight_layout()
for i in range(0, 7):
    ax[i].set(ylabel=f'{i+8}')
    
fig.suptitle('H Matrix Loadings vs. Wavelengths | NNMF')
plt.show()

# ---------
# Vectors 15-20

fig, ax = plt.subplots(6, figsize=(9,8))
for i in range(14, 20):
    ax[i - 14].plot(data.wavelengths[0][80:1200], nnmf.components_[i][80:1200], color='blue')
    ax[i - 14].spines['left'].set_position(('axes', 0.04))
    ax[i - 14].spines['bottom'].set_position(('axes', -0.05))
    ax[i - 14].spines['top'].set_visible(False)
    ax[i - 14].spines['right'].set_visible(False)

fig.tight_layout()
for i in range(0, 6):
    ax[i].set(ylabel=f'{i+15}')
      
fig.suptitle('H Matrix Loadings vs. Wavelengths | NNMF')
plt.show()