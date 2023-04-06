# Data Science project - Loan Predicton System

##Understanding the Dataset##

Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.

This is a standard supervised clasication task.A classification problem where we have to predict whether a loan would be approved or not. Below is the dataset attributes with description.

# Loan_ID : Unique Loan ID

# Gender : Male/ Female

# Married : Applicant married (Y/N)

# Dependents : Number of dependents

# Education : Applicant Education (Graduate/ Under Graduate)

# Self_Employed : Self employed (Y/N)

# ApplicantIncome : Applicant income

# CoapplicantIncome : Coapplicant income

# LoanAmount : Loan amount in thousands of dollars

# Loan_Amount_Term : Term of loan in months

# Credit_History : Credit history meets guidelines yes or no

# Property_Area : Urban/ Semi Urban/ Rural

# Loan_Status : Loan approved (Y/N) this is the target variable

Importing Basic Modules

import numpy as np #NumPy contains a multi-dimensional array and matrix data structures.
import pandas as pd #If you work with tabular, time series, or matrix data, pandas is your go-to Python package. 
import seaborn as sns #Seaborn is a high-level interface for drawing attractive statistical graphics with just a few lines of code.
import matplotlib.pyplot as plt #Matplotlib is the most common data exploration and visualization library.
#You can use it to create basic graphs like line plots, histograms, scatter plots, bar charts, and pie charts.
%matplotlib inline #magic function the 'inline' backend, your matplotlib graphs will be included in your notebook, next to the code. 
