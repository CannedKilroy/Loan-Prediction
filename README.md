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


Establishing a baseline:

- ***Logistic Regression***: Logistic regression will serve as a good baseline due to its explainability and simplicity

### Dataset Description
- The dataset, from LendingClub, covers peer to peer loans made on the Lendingclub marketplace. It includes loans made
from 2007 to 2018, encompassing a wide range of features including applicant demographics, loan specifics, loan outcomes.  

To run the program:
- Download and extract the csv's from:
https://www.kaggle.com/datasets/wordsforthewise/lending-club
then place both csv's inside /Data/Lending_club

- Accepted_EDA.ipnyb for accepted loans EDA (work in progress)
- data_dict.ipnyb to scrape for the data dictionary (work in progress)


### Dependencies
---

**Python Version**: 

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





### Results
---

The best model amongst the ones tried so far is

- **Baseline Model 1 (Logistic Regresssion)**


### Next Steps
---
- Try to standardize employee title using regex or NLP

### Licensing, Authors, Acknowledgements
---
The data for this analysis has come from:

- [LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [LendingClub Data Dictionary](https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary)




