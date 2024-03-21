import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import resample
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import BaseEstimator


from tempfile import mkdtemp
from pathlib import Path
from joblib import dump
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ######################################################################################
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def data_dict():
    '''
    Display the data dict
    '''
    #pathlib is used to ensure compatibility across operating systems
    try:
        data_destination = Path('../Data/Lending Club Data Dictionary Approved.csv')
        dict_df = pd.read_csv(data_destination, encoding='ISO-8859-1')
        display(dict_df.iloc[:,0:2])
    except FileNotFoundError as e:
        print(e.args[1])
        print('Check file location')

# ######################################################################################
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

# ######################################################################################
class Model_Wrapper:
    
    def __init__(self, df):
        '''
        Pass the full cleaned df in. 
        
        '''
        
        self.df = df
        self.result_cache = {}
        #self.model = model
        self.preprocessor = None
        self.random_state = 1


    def train_test_split(self, target_column, *args, **kwargs):
        # Split the data
        X = self.df.drop(columns=[target_column], inplace=False)
        y = self.df[target_column]       
        
        return train_test_split(X, y, random_state = self.random_state, stratify=y,  *args, **kwargs)

    def display_corr_heatmap(self, df):
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

    
    
    def multi_collinearity(self, vif_cutoff = 10):
        '''
        Retruns the columns with high multi_collinearity. 
        '''
        # Select only numeric features
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Create a dataframe to hold the vif scores for each feature
        vif_data = pd.DataFrame()
        vif_data['feature'] = numeric_df.columns
        
        # Calculate the vif
        vif_data['VIF'] = [variance_inflation_factor(numeric_df.values, i) for i in range(len(numeric_df.columns))]
        
        vif_data.sort_values(by=['VIF'], ascending=False)
        
        high_vif_columns = vif_data[vif_data['VIF'] > vif_cutoff]['feature'].tolist()
        
        return high_vif_columns
    
    def balance_data(self, X_train, y_train):
        '''
        Balances a dataset by downsampling the class 1. It is assumed the target variable is encoded in 1 and 0 test. 
        '''
        print('Number of class 1 examples before:', X_train[y_train == 1].shape[0])        
        
        # Downsample majority class without replacement to the same size of the minority class
        X_downsampled, y_downsampled  = resample(X_train[y_train == 1],
                                           y_train[y_train == 1],
                                           replace = False,
                                           n_samples = X_train[y_train == 0].shape[0],
                                           random_state = self.random_state)
        
        print('Number of class 1 examples after:', X_downsampled.shape[0])        
        
        # Combine the downsampled successful loans with the failed loans
        X_train_bal = pd.concat([X_train[y_train == 0], X_downsampled])
        y_train_bal = np.hstack((y_train[y_train == 0], y_downsampled))
        
        print("New X_train shape: ", X_train_bal.shape)
        print("New y_train shape: ", y_train_bal.shape)
        return X_train_bal, y_train_bal
        
    # ##################################################################################################
    def setup_transformer(self, transformers, n_jobs = 2):
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough', n_jobs=n_jobs)
        return self.preprocessor

    def fit_transformer(self, X_train):
        self.preprocessor.fit(X_train)
        
    def transform_data(self, X):
        return self.preprocessor.transform(X)
    
    def save_preprocessor(self, name):
        '''
        Name of the preprocessor
        '''
        preprocessor_path = Path(f'../Models/{name}.joblib')
        dump(self.preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
    # ##################################################################################################

    def run_model(self, model: BaseEstimator, X_train, y_train, *args, **kwargs):
        """
        Fit a given model to the training data.

        Parameters:
        model (BaseEstimator): The model to be trained.
        X_train: Training data features.
        y_train: Training data target variable.
        *args, **kwargs: Additional arguments to pass to the model's fit method.
        """
        model.fit(X_train, y_train, *args, **kwargs)
        self.model = model
        print("Model training complete.")
        
    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        X_test: Test data features.
        
        Returns:
        Predictions made by the model.
        """
        if self.model:
            return self.model.predict(X_test)
        else:
            raise Exception("Model not trained. Please run 'run_model' first.")
    # ##################################################################################################

    def evaluate_confusion_matrix(self, y_test, y_pred):
        # Evaluating the model with confusion matrix and a classification report
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Display the confusion matrix
        ConfusionMatrixDisplay.from_estimator(
            log_reg, 
            X_test_transformed, 
            y_test, 
            cmap='Blues', 
            display_labels=['Class 0', 'Class 1']
        )
        plt.title('Confusion Matrix for Logistic Regression')
        plt.show()
        
        print("Confusion matrix:")
        print(conf_matrix)
        
    def evaluate_classification_report(self, y_test, y_pred):
        class_report = classification_report(y_test, y_pred)

    def run_iteration(self,):
        pass