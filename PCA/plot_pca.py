"""
Zachariah Kline
31 OCT 2022
"""

from pca import pca, transformed, transformed_train, transformed_test
import matplotlib.pyplot as plt


def plot_trained(pca_obj, train, test, indep_v=0, dep_v=1, plot_test_values=True):
    """
    Plots the splitted/transformed data in a reduced (by PCA) two dimensional representation.\n

    Args:\n
        pca_obj (class): Holds statistical information about the reductions.\n
        train (numpy.ndarray): Training data that was dimensioanlly reduced by PCA \n
        test (numpy.ndarray): Testing data that was dimensioanlly reduced by PCA \n
        indep_v (int, optional): What principal component you want on the x-axis. Defaults to 0.\n
        dep_v (int, optional): What principal component you want on the y-axis. Defaults to 1.\n
        plot_test_values (bool, optional): Values that were fitted by the trained data plotted with train values. Defaults to True.
    """
    if plot_test_values:
        for row in test:
            plt.scatter(row[indep_v], row[dep_v], color='red', marker='.', zorder=2)
        for row in train:
            plt.scatter(row[indep_v], row[dep_v], color='grey', marker='o') # Grey to make it easier to see.
    else:
        for row in train:
            plt.scatter(row[indep_v], row[dep_v], color='blue', marker='.')
        
    plt.grid()
    # explained_variance_ratio gives the value associcated with how much of the original data variance is being represented
    # inside of that particular principal component.
    plt.xlabel(f'Principal Component {indep_v + 1} - ({round(pca_obj.explained_variance_ratio_[indep_v] * 100, 2)}%)')
    plt.ylabel(f'Principal Component {dep_v + 1} - ({round(pca_obj.explained_variance_ratio_[dep_v] * 100, 2)}%)')
    plt.title('2D Projections - PCA')
    plt.show()
    
    
def plot_all(pca_obj, data, indep_v=0, dep_v=1):
    count = 0
    co_count = 0
    # These colors can be changed to whatever, but 11 unique colors required.
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'black', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for row in data:
        count += 1
        plt.scatter(row[indep_v], row[dep_v], color=colors[co_count], marker='.')
        if count % 100 == 0 and count != 0:
            co_count += 1
        
    plt.grid()
    plt.xlabel(f'Principal Component {indep_v + 1} - ({round(pca_obj.explained_variance_ratio_[indep_v] * 100, 2)}%)')
    plt.ylabel(f'Principal Component {dep_v + 1} - ({round(pca_obj.explained_variance_ratio_[dep_v] * 100, 2)}%)')
    plt.title('2D Projections - PCA')
    plt.show()
    
    

if __name__ == '__main__':
    plot_trained(pca, transformed_train, transformed_test, 0, 1, True)
    plot_trained(pca, transformed_train, transformed_test, 0, 1, False)
    plot_all(pca, transformed, 0, 1)