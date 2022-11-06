"""
Zachariah Kline
Updated: 6 Nov 2022
"""

from matplotlib.colors import CSS4_COLORS
from data import Data
from numpy import mean
import matplotlib.pyplot as plt

def plot(data, row, x1lim=0, x2lim=0):
    """
    Plots one row from the absorb_measurements_pos matrix.\n

    Args:\n
        row (int): Row selector (0-1099).\n
        x1lim (int, optional): Left crop in nanometers. Defaults to 0 (no constraint on x).\n
        x2lim (int, optional): Right crop in nanometers. Defaults to 0 (no constraint on x).
    """
    HNO3 = round(data.concentrations[row][0], 5)
    Pu4 = round(data.concentrations[row][1], 5)
    
    plt.plot(data.wavelengths[0], data.absorb_measurements_pos[row], label=f'Pu4 = {Pu4} (M)\nHNO3 = {HNO3} (M)', color='blue')
    
    if x1lim == 0 and x2lim == 0:
        #Plots the entire spectra.
        plt.xlim(data.wavelengths[0][0], data.wavelengths[0][-1])
    else:
        #Plots the desired nanometer crop.
        plt.xlim(x1lim, x2lim)
        
    plt.legend()
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Absorbance Level')
    plt.title(f'Absorption Measurements For Row {row + 1}')
    plt.grid(axis='y')
    plt.show()
    
def plot_all(data, all=False, x1lim=0, x2lim=0):
    """
    Plots all absorption measurements from the absorb_measurements_pos matrix that\n
    correspond to a unique concentration pair of Pu4 and HNO3 (110 lines or 1100).\n

    Args:\n
        all (bool, optional): If True, the function will plot every row. Otherwise will average group concentrations.\n
        x1lim (int, optional): Left crop in nanometers. Defaults to 0 (no constraint on x).\n
        x2lim (int, optional): Right crop in nanometers. Defaults to 0 (no constraint on x).
    """
    colors = [color for color in CSS4_COLORS.keys()]
    count = 0
    if all:
        for row in data.absorb_measurements_pos:
            plt.plot(data.wavelengths[0], row, color=colors[count])
            count += 4
            
            if count >= len(colors):
                count = 0
                
        plt.title('Absorbance Measurements | All')
    else:
        #Only plots a row when the concentration pair (HNO3, Pu4) changes.
        count = 0
        for i in range(0, 1100, 10):
            plt.plot(data.wavelengths[0], mean(data.absorb_measurements_pos[i:i+10], axis=0), color=colors[count])
            count += 4
            
            if count >= len(colors):
                count = 0
                
        plt.title('Absorbance Measurements | Averaged Grouping')
    
    if x1lim == 0 and x2lim == 0:
        #Plot the entire spectra.
        plt.xlim(data.wavelengths[0][0], data.wavelengths[0][-1])
    else:
        #Plots the desired nanometer crop.
        plt.xlim(x1lim, x2lim)
        
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Absorbance Levels')
    plt.grid(axis='y')
    plt.show()
    
def plot_concentrations(data):
    """
    Plots the concentrations of the samples that were measured for the absorb_measurements_pos matrix.\n
    NOTE: Every dot represents a 10 replications of that concentration.
    """
    for HNO3, Pu4 in data.concentrations:
        plt.plot([HNO3], [Pu4], '2', color='red')
    
    plt.xlabel('HNO3 (M)')
    plt.ylabel('Pu4 (M)')
    plt.title('Sample Concentrations')
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    data = Data(split_type='rand', training_size=0.8)
    # Examples of use.
    plot(data, 128)
    plot_all(data, True)
    plot_concentrations(data)