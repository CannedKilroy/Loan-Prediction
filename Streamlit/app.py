import streamlit as st
from joblib import load
import pandas as pd
from pathlib import Path

# ############# LOAD THE DATA #####################
parquet_file_path = Path('Cleaned_Data/model_cleaned')
try:
    # Read the parquet file
    loans_df = pd.read_parquet(parquet_file_path)
except FileNotFoundError as e:
    print(e.args[1])
    print('Check file location')
    
# ############# LOAD THE MODELS #####################
model_path = Path('Models/best_logistic_regression_model.joblib')
model = load(model_path)

# ############# CHOOSE RANDOM LOAN  #####################
loan = loans_df.iloc[[3]].copy()

# ############# DEFINE THE LAYOUT #####################
st.image('Streamlit/banner_2.png', use_column_width=True)
st.title('Loan Default Prediction')

st.write('''
Welcome to the interactive demo!
This app showcases a logistic regression 
model for predicting whether a loan will be successful, 
designed for demonstration purposes. Due to the number of features used in the model, only 
6 were chosen to be interactive for simplicity. Note that the remaining feature values 
are hard coded, meaning any changes might have a small effect on the predicted probability,
and that a probability threshold of 0.5 is used.
''')

# ############# USER INPUTS #####################

#Loan amount
loan_amount = st.slider('Loan Amount', min_value=0, max_value=40000, value=10000, step=500) # default value
loan['loan_amnt'] = loan_amount

# Employment Length
emp_length = st.slider('Employment Length of last job', min_value=0, max_value=10, value=2, step=1)
loan['emp_length'] = emp_length

# Home ownership
ownership_options = ['MORTGAGE', 'RENT', 'OWN', 'ANY', 'NONE', 'OTHER']
home_ownership = st.selectbox('Home Ownership Status', ownership_options, index=1)
loan['home_ownership'] = home_ownership

# Loan Purpose
purpose_options = ['debt_consolidation', 'vacation', 'credit_card', 'home_improvement', 
                   'other', 'major_purchase', 'car', 'house', 
                   'moving', 'small_business', 'medical', 'renewable_energy', 'wedding']
purpose = st.selectbox('Select Loan Purpose', purpose_options, index=0)
loan['purpose'] = purpose

# Annual Income
income = st.slider('Annual Income', min_value=0, max_value=200000, step=1000, format='%f', value = 40000)
loan['annual_inc'] = income

#Interest Rate
rate = st.slider('Interest Rate', min_value = 5.0, max_value = 31.0, step = 0.05, value = 18.0)
loan['int_rate'] = rate

# ############# DROP COLUMNS #####################
columns_to_drop = ['installment', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 
 'tot_cur_bal', 'total_rev_hi_lim', 'bc_open_to_buy', 'bc_util', 
 'mort_acc', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 
 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'pct_tl_nvr_dlq',
 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
 'num_tl_op_past_12m', 'loan_status']
loan.drop(columns=columns_to_drop, inplace=True)
processed_input = loan

# ############# PREDICT #####################
if st.button('Predict Default'):
    
    probabilities = model.predict_proba(processed_input)
    
    # Get the probability for the loan being successful (class 1)
    probability_success = probabilities[0, 1]
    
    # Get the probability for the loan defaulting (class 0)
    probability_default = probabilities[0, 0]
    
    # Display the probabilities
    st.write(f'Probability of loan being paid off successfully: {probability_success:.2f}')
    st.write(f'Probability of loan defaulting: {probability_default:.2f}')
    
    if probability_success > 0.50:
        st.image('Streamlit/approved.png')
    else:
        st.image('Streamlit/denied.jpg')    