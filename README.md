# Classification - Loan Data Analysis

### Project Overview
With an estimated size of $153 billion dollars in 2022, the peer to peer (P2P) loan market is a viable alternative to traditional lending institutions. However, with the challenges of collateralization and information asymetry in the P2P markets, how can investors properly assess applicants credit worthines?

The primary objective of this project is to use both classical statistical methods such as Logistic Regression, and more complex machine learning models such as:
- ***Decision Trees***
- ***Random Forests***
- ***XGBoost***
- ***K-Nearest Neighbors (KNN)***

to ***predict whether the borrower will default on their loan***

The secondary objective is to explore ***whether a similar accuracy is achievable without the use of grey metrics such as fico scores.***


Baseline has been established using two methods -

- **Naïve Forecasting Method**: Since we intend to provide short-term forecasts, our first baseline model will assume that the predicted value at time `t` will be equal to the actual value of demand at time `t-1`.
- **Seasonal Decomposition**: In our second baseline model we extract the trend and seasonalities from our training data and use them to forecast demand.

The Jupyter Notebooks in this repository have been labelled stepwise and should be followed in order -

- step_1_data_preparation.ipynb
- step_2_exploratory_data_analysis.ipynb
- step_3_modeling.ipynb


### Dependencies
---
**OS**: macOS Ventura 13.4.1

**Python Version**: 3.8.10

**Python Libraries** (please refer requirements.txt):
- pandas
- numpy
- statsmodels
- scipy
- scikit-learn
- matplotlib
- seaborn
- plotly
- xgboost
- joblib


### Dataset Description
---
This has been moved under the "Data" folder.


### Results
---

The best model amongst the ones tried so far is `XGBoost` - and by a huge margin.

- **Baseline Model 1 (Naïve model)**
	- MAPE = 0.278 %
	- RMSE = 2710

- **Baseline Model 2 (Seasonal decomposition based)**
	- MAPE = 0.304 %
	- RMSE = 2672

- **XGBoost**

    (`colsample_bytree`: 0.55, `learning_rate`: 0.24, `max_depth`: 2, `n_estimators`: 293, `reg_alpha`: 0.3)
	- MAPE = 0.178 %
	- RMSE = 1970

- **ARIMA (1,0,7)**
    
    (data = `total_demand` differenced twice - 365,7)
	- MAPE = 0.38 %
	- RMSE = 3433

- **SARIMA (4,0,0) (0,0,1,7)**
    
    (data = log of `total_demand` differenced once - 365)
	- MAPE = 0.4 %
	- RMSE = 3451

### Next Steps
---
- There is a lot of repetitive code in this notebook, create classes/functions where possible and clean this notebook up

- Investigate the outliers in bike rental demand data, check if the spikes in the data correspond to promotional events or any other such critical information which may help inform data preparation

- Investigate if seasonal decomposition can be used effectively to remove multiple seasonalities (consider using [statsmodels-MSTL](https://www.statsmodels.org/dev/examples/notebooks/generated/mstl_decomposition.html)) from the data before employing (S)ARIMA

- Explore RNNs like LSTM to forecast demand and compare with XGBoost

### Licensing, Authors, Acknowledgements
---
The data for this analysis has been curated from the following sources:

- [Capital Bike Share](https://www.capitalbikeshare.com/system-data): this includes the bike demand data for the metro DC area
- [NOAA's National Climatic Data Center](https://www.ncdc.noaa.gov/cdo-web/search): this includes the weather data for the metro DC area
- [DC Department of Human Resources](https://edpm.dc.gov/issuances/legal-public-holidays-2023/): this includes the holiday data for Washington, D.C.




