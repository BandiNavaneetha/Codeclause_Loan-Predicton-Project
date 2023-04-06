# Data Science project - Loan Predicton System

##Understanding the Dataset##

Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.

This is a standard supervised clasication task.A classification problem where we have to predict whether a loan would be approved or not. Below is the dataset attributes with description.

Loan_ID : Unique Loan ID
Gender : Male/ Female
Married : Applicant married (Y/N)
Dependents : Number of dependents
Education : Applicant Education (Graduate/ Under Graduate)
Self_Employed : Self employed (Y/N)
ApplicantIncome : Applicant income
CoapplicantIncome : Coapplicant income
LoanAmount : Loan amount in thousands of dollars
Loan_Amount_Term : Term of loan in months
Credit_History : Credit history meets guidelines yes or no
Property_Area : Urban/ Semi Urban/ Rural
Loan_Status : Loan approved (Y/N) this is the target variable

**Importing Basic Modules #pandas#numpy#matplotlib#seaborn#scikit-learn
Import Datasets #read csv file**

##optional
Display Top 5 Rows of The Dataset 
Check Last 5 Rows of The Dataset
Find Shape of Our Dataset (Number of Rows And Number of Columns)
Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement
Finding duplicate values
Check Null Values In The Dataset

#Exploratory Data analysis
analys the data for each categorica attribute #categorica attribute visulization
from this alaysis we will getting some intuition, useful to build model
numercial attribues visualization
mean is in middle
##Data Cleaning
Handling The missing Values
Handling Categorical Columns
Store Feature Matrix In X And Response (Target) In Vector y
Feature Scaling
Splitting The Dataset Into The Training Set And Test Set & Applying K-Fold Cross Validation

#Cheching Accuracy
Logistic Regression
SVC(Support vector machines)
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier

#checking which classification is best 
Hyperparameter Tuning
Random Forest Classifier

#Model building
Save The Model
GUI
