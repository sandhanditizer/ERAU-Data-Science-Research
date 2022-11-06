"""
Zachariah Kline
Updated: 6 Nov 2022
"""

from pca import pca, data, transformed_train, transformed_test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from numpy import sqrt, polyfit, median, mean
import matplotlib.pyplot as plt


def linear_regression(concen_train, concen_test, transformed_train, transformed_test):
    LR_model = LinearRegression()
    LR_model.fit(transformed_train, concen_train)
    predicted_concen = LR_model.predict(transformed_test)
    return (predicted_concen, concen_test)
    
def display_Pu(pca_obj, predicted, test):
    # Median Error (ME) is only used for understanding if the linear model is under or over predicting values.
    # If the ME is positive, then then model, on average, under predicted concentrations given this training set.
    # If otherwise negative, then the converse is true. Using median because we dont want thrown off by bad predictions.
    ME = median(test[:,1] - predicted[:,1])
    # Root Mean Squared Error (RMSE) is a more accurate indication of how off, on average, the predictions are from
    # our known test values.
    RMSE = sqrt(mean((test[:,1] - predicted[:,1]) ** 2))
    # Coefficent of Determination (r^2 score) is used to understand how well the linear model fits the mean concentrations.
    R2 = r2_score(test[:,1], predicted[:,1])
    
    error = f'Components Used: {pca_obj.n_components_}\n' 
    error += f'ME = {round(ME, 6)}\n' 
    error += f'RMSE = {round(RMSE, 4)} (M)\n'
    error += r'r$^2$ = ' + f'{round(R2, 4)}' 
        
    X = []
    Y = []
    # Plotting individual comparison points.
    for test, pred in zip(test[:,1], predicted[:,1]):
        plt.plot([test], [pred], '.', color='dimgray')
        X.append(test)
        Y.append(pred)
    
    # Best fit line.
    m, b = polyfit(X, Y, deg=1)
    f = [(m*x+b) for x in X]
    plt.plot(X, f, label=error, color='blue')

    plt.xlabel('Observed (M)')
    plt.ylabel('Predicted (M)')
    plt.title(r'Predicted vs. Observed Concentrations | Pu$^{+4}$')
    plt.grid()
    plt.legend()
    plt.show()
    
def display_HNO3(pca_obj, predicted, test):
    ME = median(test[:,0] - predicted[:,0])
    RMSE = sqrt(mean((test[:,0] - predicted[:,0]) ** 2))
    R2 = r2_score(test[:,0], predicted[:,0])
    
    error = f'Components Used: {pca_obj.n_components_}\n' 
    error += f'ME = {round(ME, 6)}\n' 
    error += f'RMSE = {round(RMSE, 4)} (M)\n'
    error += r'r$^2$ = ' + f'{round(R2, 4)}' 
        
    X = []
    Y = []
    for test, pred in zip(test[:,0], predicted[:,0]):
        plt.plot([test], [pred], '.', color='dimgray')
        X.append(test)
        Y.append(pred)
    
    m, b = polyfit(X, Y, deg=1)
    f = [(m*x+b) for x in X]
    plt.plot(X, f, label=error, color='orange')

    plt.xlabel('Observed (M)')
    plt.ylabel('Predicted (M)')
    plt.title(r'Predicted vs. Observed Concentrations | HNO$_3$')
    plt.grid()
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    # Change number of components and data constraints in pca.py file.
    predicted, test = linear_regression(data.concen_train, data.concen_test, transformed_train, transformed_test)
    display_Pu(pca, predicted, test)
    display_HNO3(pca, predicted, test)