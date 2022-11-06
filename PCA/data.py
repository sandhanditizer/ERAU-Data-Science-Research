"""
Zachariah Kline
Updated: 6 nov 2022
"""

from scipy.io import loadmat
from numpy import array
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, split_type, training_size):
        # Convert data from a *.mat format to a python dictionary.
        data_dictionary = loadmat('dataset.mat') 
        # Variables from original *.mat file are as follows:
        self.wavelengths = data_dictionary['x']
        self.concentrations = data_dictionary['conc_HNO3_Pu4']
        absorb_measurements = data_dictionary['Y_Pu4_UVvis']
        
        # ~13% of the data was negative. Since negative spectroscopy measurements
        # do not make sense, we want turn all negatives to zero.
        absorb_pos = []
        for row in absorb_measurements:
            temp_row = []
            for num in row:
                if num < 0:
                    temp_row.append(0)
                else:
                    temp_row.append(num)
            
            absorb_pos.append(temp_row)
            
        self.absorb_measurements_pos = array(absorb_pos)

        # Seperating train/test datasets
        if split_type == 'rand':
            self.absorb_train, self.absorb_test, self.concen_train, self.concen_test = train_test_split(
                                                                                    self.absorb_measurements_pos, 
                                                                                    self.concentrations, 
                                                                                    train_size=training_size)
        elif split_type == 'strat':
            self.absorb_train, self.absorb_test, self.concen_train, self.concen_test = train_test_split(
                                                                                    self.absorb_measurements_pos, 
                                                                                    self.concentrations, 
                                                                                    train_size=training_size,
                                                                                    stratify=self.concentrations)
        else:
            raise Exception('split_type must be `rand` or `strat`.')