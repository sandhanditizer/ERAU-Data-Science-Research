"""
Zachariah Kline
Updated: 6 Nov 2022
Algorithm that determines, given a tolerance r^2 score, the smallest dimension needed to achieve
accurate plutonium and nitric acid predictions.
"""

from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from numpy import mean, std, sqrt
from data import Data
from sys import stdout
import matplotlib.pyplot as plt


t_rank = 30 # Biggest rank size that the algorithm will calculate.
type = 'rand'
MoR = 20 # How many averages you want to take.
goal = 0.98 # r^2 tolerance.

# Setup for progress bar
stdout.write("[%s]" % (" " * MoR))
stdout.flush()
stdout.write("\b" * (MoR+1))

X = [i for i in range(1, t_rank+1)]
Pu_RMSE = []
Pu_R2 = []
HNO3_RMSE = []
HNO3_R2 = []

for a in range(0, MoR):
    data = Data(split_type=type, training_size=0.8)
    
    Pu_RMSE.append([0] * t_rank)
    Pu_R2.append([0] * t_rank)
    HNO3_RMSE.append([0] * t_rank)
    HNO3_R2.append([0] * t_rank)
        
    for b in range(0, t_rank):
        
        nnmf = NMF(n_components = b+1, init='nndsvd', max_iter=1000)
        
        W_train = nnmf.fit_transform(data.absorb_train)
        W_test = nnmf.transform(data.absorb_test)
        
        LR_model = LinearRegression()
        LR_model.fit(W_train, data.concen_train)
        predicted_concen = LR_model.predict(W_test)
        
        Pu_RMSE[a][b] = sqrt(mean((data.concen_test[:,1] - predicted_concen[:,1]) ** 2))
        Pu_R2[a][b] = (r2_score(data.concen_test[:,1], predicted_concen[:,1]))
        HNO3_RMSE[a][b] = sqrt((mean(data.concen_test[:,0] - predicted_concen[:,0]) ** 2))
        HNO3_R2[a][b] = (r2_score(data.concen_test[:,0], predicted_concen[:,0]))

    stdout.write("-")
    stdout.flush()

# Average out all the RMSEs.
Pu_RMSE_mean = mean(Pu_RMSE, axis=0)
Pu_R2_mean = mean(Pu_R2, axis=0)
HNO3_RMSE_mean = mean(HNO3_RMSE, axis=0)
HNO3_R2_mean = mean(HNO3_R2, axis=0)

# Computes the standard deviation.
Pu_RMSE_std = std(Pu_RMSE, axis=0)
Pu_R2_std = std(Pu_R2, axis=0)
HNO3_RMSE_std = std(HNO3_RMSE, axis=0)
HNO3_R2_std = std(HNO3_R2, axis=0)

# Plutonium RMSE plot.
plt.subplots(figsize=(9,6))
plt.plot(X, Pu_RMSE_mean, color='mediumturquoise', label='Plutonium (IV)', zorder=2)
plt.errorbar(X, Pu_RMSE_mean, yerr=Pu_RMSE_std, ecolor='indianred', label='Standard Deviation', color='mediumturquoise')
plt.xlabel('Rank Size')
plt.ylabel('Root Mean Squared Error (M)')
plt.title(f'RMSE vs. Rank - {MoR} MoR - {type}')
plt.legend()
plt.grid(axis="x")
plt.show()

# Nitric acid RMSE plot.
plt.subplots(figsize=(9,6))
plt.plot(X, HNO3_RMSE_mean, color='darkslateblue', label='Nitric Acid', zorder=2)
plt.errorbar(X, HNO3_RMSE_mean, yerr=HNO3_RMSE_std, ecolor='indianred', label='Standard Deviation', color='darkslateblue')
plt.xlabel('Rank Size')
plt.ylabel('Root Mean Squared Error (M)')
plt.title(f'RMSE vs. Rank - {MoR} MoR - {type}')
plt.legend()
plt.grid(axis="x")
plt.show()    

# Comparison R2 plots.
plt.subplots(figsize=(9,6))
plt.plot(X, HNO3_R2_mean, color='darkslateblue', label=r'HNO$_3$', zorder=3)
plt.errorbar(X, HNO3_R2_mean, yerr=HNO3_R2_std, ecolor='black', label=r'Standard Deviation HNO$_3$', color='darkslateblue')
plt.plot(X, Pu_R2_mean, color='mediumturquoise', label=r'Pu$^{+4}$', zorder=3)
plt.errorbar(X, Pu_R2_mean, yerr=Pu_R2_std, ecolor='black', label=r'Standard Deviation Pu$^{+4}$', color='mediumturquoise')
plt.xlabel('Rank Size')
plt.ylabel(r'r$^2$ Score')
plt.title(r'r$^2$ vs. Rank' + f' - {MoR} MoR - {type}')

# Marks where the nth component reaches an average r^2 tolerance goal.
Pu_Y = set([i for i in Pu_R2_mean if i < 1 and i >= goal])
HNO3_Y = set([i for i in HNO3_R2_mean if i < 1 and i >= goal])

if Pu_Y:
    Pu_X = list(Pu_R2_mean).index(Pu_Y.pop()) + 1
    plt.scatter(Pu_X, Pu_Y.pop(), color='black', label=f'{Pu_X} Components', marker='x', zorder=5)
if HNO3_Y:
    HNO3_X = list(HNO3_R2_mean).index(HNO3_Y.pop()) + 1
    plt.scatter(HNO3_X, HNO3_Y.pop(), color='black', label=f'{HNO3_X} Components', marker='x', zorder=5)
    
plt.grid(axis="x")
plt.legend()
plt.show()