#helpers.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def data_dict():
    '''
    Display the data dict
    '''
    #pathlib is used to ensure compatibility across operating systems
    try:
        data_destination = Path('../Data/Lending_club/Lending Club Data Dictionary Approved.csv')
        dict_df = pd.read_csv(data_destination, encoding='ISO-8859-1')
        display(dict_df.iloc[:,0:2])
    except FileNotFoundError as e:
        print(e.args[1])
        print('Check file location')

def display_corr_heatmap(df):
    '''
    Takes in a df, pulls out the columns that are numeric, and displays
    the half correlation matrix
    '''
    # Select only the numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate the correlation matrix
    corr = numeric_df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True)
    plt.show()