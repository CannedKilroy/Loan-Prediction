# Classification - Loan Data Analysis
### Date: Nov 9, 2023  
Streamlit Demo app: https://loan-default-predictor.streamlit.app/
### Project Overview
-- ----
With an estimated size of $153 billion dollars in 2022, the peer-to-peer (P2P) loan market is a viable alternative to traditional lending institutions for individuals seeking small loans. However, investors face challenges in predicting the likelihood of repayment or default due to the lack of borrower information in P2P markets and lack of collateralization.

The primary objective of this project is to **predict whether a borrower will repay or default on their loan, based on the characteristics of the loan and the borrower's financial standing**, allowing investors to better minimize credit risk.

The secondary objective is to explore **whether a comparable accuracy is achievable without the use of opaque metrics such as FICO scores.** This allows for more inclusive and fair lending practices, as FICO scores can sometimes reflect biases. Furthermore, it would increase accessibility to those who have a limited credit history or are otherwise misrepresented by traditional credit scoring methods, allowing investors to extend more loans. 

### Methods and Models Used
-- ----
This project utilizes several Python libraries and machine learning models to analyze loan data:
- Pandas
- Numpy
- Matplotlib
- Sci-kit learn
- joblib
- pathlib
- sklearnx

Classical statistical methods, such as Logistic Regression, and more complex machine learning models will be used, including:  
- SVM
- Decision Trees
- Random Forests
These models were chosen as they operate off different principles: SVM is distance-based, Logistic Regression probability-based, and decision trees rule based. 



### Dataset Description
-- ----
* The LendingClub dataset covers P2P loans made on the Lendingclub marketplace from 2007 to 2018. It includes information on the loan, such as loan amount and loan status, as well as applicant information. The dataset is approximately 2 million rows, with 151 columns, containing text and numeric data. For more information on the dataset, the full data dictionary can be found here:  
[LendingClub Data Dictionary](https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary)  
* More information on LendingClub:   
[LendingClub Wikipedia](https://en.wikipedia.org/wiki/LendingClub)

### Running the Project
-- -----
Create the environment:
1. `conda create --name loans_capstone python=3.11`
2. `conda activate loans_capstone` 
3. `pip install -r requirements.txt`   
4. Create a kernel (optional)  
`ipython kernel install --name "loans_capstone" --user` 

Running the project:
1. clone the repository using:
`git clone https://github.com/CannedKilroy/Loan-Evaluations.git`

2. pip install the requirements found in the requirements.txt file using:  
`pip install -r requirements.txt`

3. Create a `Data` directory inside your repo and then create a `Lending_club` folder inside the `Data` folder. All the necessary files to run all notebooks, including the data, data dict, and cleaned data files for the models, can be found in this google drive:  
[Data](https://drive.google.com/drive/folders/1-oJ72rJPTO9L9zE19ICn9jgmrbjptquR?usp=sharing)  
Download and extract the files. Place them inside the new folder:  
"/Data/Lending_club/"

4. Then run the notebooks in order:

- data_cleaning  
- eda  
- log_reg  
- svm  
- tree_based  

### Results
---
Accuracy, precision, and recall metrics will be used to evaluate a models performance. The models were however tuned for precision.   
  
The results for class 1 successful loans by optimized model:
- ***Logistic Regression***:  
Precision: 0.87  
Recall: 0.66  
F1 score: 0.75  
AUC: 0.71  
- ***SVM***  
Precision: 0.88  
Recall: 0.61  
F1 score: 0.72  
AUC: No probability estimates by Sklearn  
- ***Decision Tree***  
Precision: 0.87  
Recall: 0.64  
F1 score: 0.74  
AUC: 0.68  
- ***Random Forest***   
Precision: 0.88  
Recall: 0.61  
F1 score: 0.72  
AUC: 0.71  
  
### Steps so far
---
***Data Cleaning***
- The data so far has been rudimentarily cleaned. As each loan acts as essentially a barcode, with a specific combination values effecting the classification, missing data cannot easily be imputed. This makes cleaning the data especially difficult. As the majority of missing values are found in loans with special conditions, the simplest solution is to drop the problematic rows and columns. These special condition rows and associated columns account for only a small subset of the data. Furthermore, any columns that can leak the outcome of the loan, for example those that track loan repayments once the loan has been given, have been removed, along with any irrelevant columns such as LendingClubs internal loan tracking id's.

***EDA***
- In performing EDA, some key insights were made. The data is heavily inbalanced, with  ~ 80% of loans being successful and 20% failed. There seems to be an inverse relationship between the interest rate and the number of loans issued. This would be a good topic to further exploration, augmented with external economic data. In addition, we found that the largest purpose was for the loans was for debt_consolidation. Most interestingly, there is a linear pattern between annual income and the loan amount. The difference in interest rates for failed and successful loans was quite stark, added context when the interest rate was found to be the biggest discriminator in the decision tree. 

***Logistic Regression***
- A baseline model was fit in 3 iterations. It was found that there are many non linear relationship, however, the model provided a good benchmark, achieving ~ 87% precision for class 1 successful loans. 

***SVM***
- The optimized SVM model (non linear kernel) achieved a slightly higher precision. However, there was some overfitting. 

***Decision Tree***
- The decision had slightly lower performance metrics compared to the baseline Logistic Regression model. There were many issues with overfitting. 

***Random Forest***
- The Random Forest performed the best, achieving a 88% accuracy. 

### Next Steps
- Make streamlit app public and expand to include EDA
- Evaluate the models stability
- Interest rate was by far the best predictor for decision trees in terms of feature importance and SHAP value. Although interest rate is not definitively leaky feature, this should be explored further.  
- Further feature engineering.
- Further exploration of random forest and why it did not perform better. 
- Other classifiers like KNN
- Revise data cleaning steps such that older loans can be kept, regardless of the nulls from api changes. 
- Further exploration of common attributes among false positives


### Future Steps
---
- Standardize employee title using regex or NLP
- Standardize loan description given by the borrower using regex or NLP
- Re-Introduce the geographical features by using 3rd party information such as average or median income for a specified state or zip code
- Use cloud computing to carry out analysis on the full dataset rather than a random sample. 


### Licensing, Authors, Acknowledgements
---
The data for this analysis has come from:

- [LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [LendingClub Data Dictionary](https://www.kaggle.com/datasets/jonchan2003/lending-club-data-dictionary)
- [Backup of the data](https://drive.google.com/file/d/1CYaYaKzeQrOOwZZKOESNyzsPPOnCdE8x/view?usp=sharing)
