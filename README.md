# Classification - Loan Data Analysis
### Date: Nov 9, 2023  
### Project Overview
-- ----
With an estimated size of $153 billion dollars in 2022, the peer to peer (P2P) loan market is a viable alternative to traditional lending institutions for individuals seeking small loans. However, investors face challenges in predicting the likelihood of repayment or default due to the lack of borrower information in P2P markets and lack of collateralization.

The Primary objective of this project is to **predict whether a borrower will repay or default on their loan, based on the charecteristics of the loan and the borrowers financial standing**, allowing investors to better minimize credit risk.

The secondary objective is to explore ***whether a comparable accuracy is achievable without the use of opaque metrics such as FICO scores.***. This could lead to more inclusive and fair lending practices, as FICO scores can sometimes reflect biases. Furthermore, it would increase accessability to those who have a limited credit history or are otherwise misrepresented by tradiontal credit scoring methods, enabling investors to extend more loans. 

Classical statistical methods, such as Logistic Regression, and more complex machine learning models will be used, including:  
- Decision Trees
- Random Forests
- XGBoost
- K-Nearest Neighbors (KNN)


### Dataset Description
-- ----
* The LendingClub dataset covers P2P loans made on the Lendingclub marketplace from 2007 to 2018. It covers both information on the loan, including loan amount and loan status, as well as applicant information. The dataset is approximatly 2 million rows, with 151 columns, containing text and numeric data. The full data dictionary can be found here:  
[LendingClub Data Dictionary](https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary)  
* More information on LendingClub can be found here:   
[LendingClub Wikipedia](https://en.wikipedia.org/wiki/LendingClub)

### Running the project
The project is run in jupyter notebooks using python, using the common data science librarys such as pandas, numpy, malplotlib, seaborn, etc. 
-- -----
1. Clone or download the repository
2. pip install the requirements found in the requirements.txt file using:  
`pip install -r requirements.txt`
3. Download and extract the csv's from:  
[LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
4. Place `accepted_2007_to_2018Q4.csv` inside /Data/Lending_club/
5. Download and extract the csv's from:  
[LendingClub Data Dictionary](https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary)
6. Place `Lending Club Data Dictionary Approved.csv` inside /Data/Lending_club/

Then run the notebooks in order:
1. data_cleaning
2. eda
3. log_reg

The cleaned parquet file generated by the data_cleaning notebook can be found here:  
[Cleaned data file](https://drive.google.com/file/d/1NA3QfiQBhkoaCI89pVCxbn5FvwmI5EOS/view?usp=sharing)

### Results
---
Accuracy, precision, and recall metrics will be used to evaluate a models performance. 
- ***Logistic Regression***: Logistic regression will serve as a good baseline due to its explainability and simplicity

### Steps so far

***Data Cleaning***
- The data so far has been rudimentarily cleaned. As each loan acts as essentially a barcode, with a specific combination values effecting the classification, missing data cannot easily be imputed. This makes cleaning the data especially difficult. As the majority of missing values are found in loans with special conditions, the simplest solution is to drop the problomatic rows and columns. These special condition rows and asociated columns account for only a small subeset of the data. For example, hardship loans and joint applicant loans. Furthermore, any columns that can leak the outcome of the loan, for example those that track loan repayments once the loan has been given, have been removed, along with any irrelavent columns such as LendingClubs internal loan tracking id's.

***EDA***
- In performing EDA, some key insights were made. The data is heavily inbalanced, with  ~ 80% of loans being successful and 20% failed. This will have to be compensated for when performing the modeling. Furthermore, we found that there seems to be an inverse relationship between the interest rate and the number of loans issued. This would be a good topic to further exploration. In addition, we found that the largest purpose was for debt_consolidation. Most interestingly, there is a linear pattern between annual income and the loan amount. 

### Next Steps

- ***Further Feature Engineering***

- ***Logistic Regression***

### Future Steps
---
- Standardize employee title using regex or NLP
- Standardize loan description given by the borrower using regex or NLP
- Re-Introduce the geographical features by using 3rd party information such as average or median income for a specified state or zip code
- 


### Licensing, Authors, Acknowledgements
---
The data for this analysis has come from:

- [LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [LendingClub Data Dictionary](https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary)
- [Backup of the data](https://drive.google.com/file/d/1CYaYaKzeQrOOwZZKOESNyzsPPOnCdE8x/view?usp=sharing)