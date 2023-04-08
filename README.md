 **Data Science project - Loan Predicton**
 
**##Understanding the Dataset##**
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

**Import Libraries**  
import numpy as np: This imports the numpy library and gives it the alias np, which is commonly used in Python data analysis.
import pandas as pd: This imports the pandas library and gives it the alias pd. Pandas is a powerful library for data manipulation and analysis.
import seaborn as sns: This imports the seaborn library, which provides a high-level interface for creating informative and attractive statistical graphics.
from matplotlib import pyplot as plt: This imports the pyplot module from the matplotlib library and gives it the alias plt. Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
%matplotlib inline: This is a magic command for Jupyter Notebook that enables Matplotlib to display plots inline in the notebook.
import warnings: This imports the warnings module, which allows Python to issue warning messages.
warnings.filterwarnings('ignore'): This line suppresses warning messages so that they are not displayed in the output.

**Importing Datasets** 
Read in data from two CSV files named "train.csv" and "test.csv" into pandas DataFrames named df and tf, respectively.
pd.read_csv() is a pandas function that reads in CSV files and creates a DataFrame object. The file path is passed as an argument to the function. In this case, the files "train.csv" and "test.csv" should be in the current working directory or the specified file path.
By convention, "train.csv" usually contains the training data for a machine learning model, while "test.csv" contains the testing data. Once the data is loaded into pandas DataFrames, it can be easily manipulated, cleaned, and analyzed using pandas functions and other Python libraries.

**Understanding the data:**
Displaying top 10 rows of the Dataset
Displaying last 10 rows of dataset
Find Shape of Our Dataset (Number of Rows And Number of Columns)

**Preprocessing the Dataset:
perform data cleaning and imputation of missing values in the df DataFrame 
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean()): This line fills in the missing values in the "LoanAmount" column with the mean value of the column. This is a common imputation method for continuous variables.
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0]): This line fills in the missing values in the "Gender" column with the mode (most frequent value) of the column. This is a common imputation method for categorical variables.

**Exploratory Data analysis:**
Analyzing the data for each categorical attribute
sns.countplot(): This is a seaborn function for creating count plots.

**Creation of new Attributes:**
Total_amount based Applicant amount and Coapplicant Amount

**Log Transformation:**
"ApplicantIncomeLog" that contains the natural logarithm of the "ApplicantIncome" column plus one. The purpose of adding one is to avoid taking the logarithm of zero, which is undefined. This is a common transformation used in data analysis and modeling to normalize the distribution of a variable that is positively skewed.
creates a histogram of the "ApplicantIncomeLog" column using the seaborn library's distplot() function. The distplot() function combines a histogram of the data with a kernel density estimate (KDE), which is a non-parametric way to estimate the probability density function of a random variable.
 distribution of the "ApplicantIncomeLog" variable, which is now closer to a normal distribution than the original "ApplicantIncome" variable.
 
**Coorelation Matrix:**
calculates the correlation matrix for all numerical columns in the df DataFrame using the corr() method. This creates a square matrix where each cell contains the correlation coefficient between the two corresponding columns.creates a figure with a size of 12 by 8 inches using the figure(figsize=(12,8)) function from matplotlib. This sets the size of the figure to be larger than the default size.creates a heatmap plot using the seaborn library's heatmap() function. The heatmap plot visualizes the correlation matrix calculated earlier. The annot=True argument adds the correlation coefficient values to each cell in the plot. The cmap="BuPu" argument sets the color map for the plot to a blue-purple gradient.The resulting heatmap plot shows the correlation matrix for the numerical columns in the df DataFrame. 

**Label Encoding:**
Imports the LabelEncoder class from the sklearn.preprocessing module and uses it to transform categorical columns in the df DataFrame into numerical labels.
The cols variable is a list that contains the names of the columns that need to be transformed. These columns are 'Gender', "Married", "Education", 'Self_Employed', "Property_Area", "Loan_Status", and "Dependents".
The LabelEncoder class provides a simple way to encode categorical values as integers. The fit_transform() method of the LabelEncoder class fits the encoder on the unique values of a column and transforms the values into a numerical label. The numerical labels are assigned in alphabetical order, so the labels for the same category will be the same across different columns.
The for loop iterates over the columns in the cols list, applies the LabelEncoder transformation to each column, and assigns the transformed values back to the corresponding column in the df DataFrame.
The resulting DataFrame will have transformed the categorical variables into numerical labels, which can be used in some machine learning models that require numerical inputs.

**Train-Test Split:**
Train-test split on the data. It splits the dataset X and target variable y into training and testing sets, with 75% of the data used for training the model and 25% of the data used for testing the model. The parameter random_state is used to ensure that the same random data points are selected for the split each time the code is run, making the results reproducible. The split data is stored in four variables, x_train, x_test, y_train, and y_test, which are used for training and testing the model.

**Model Training:**
A function called classify() which takes in a machine learning model (e.g., RandomForestClassifier), feature matrix x, and target vector y as input. The function then splits the data into training and testing sets using train_test_split() function from scikit-learn with a test size of 0.25 and a random seed of 42. The model is then trained on the training data using the fit() method, and its accuracy is computed on the testing data using the score() method. Additionally, cross-validation is performed using the cross_val_score() function from scikit-learn with k-fold cross-validation (k=5) and the mean cross-validation accuracy is printed. The purpose of this function is to train and evaluate the model with both train-test split and cross-validation to ensure the model is robust and generalizable. Checking the best accuracy given classifier.

**Hyperparameter tuning:**
Initializes a Random Forest Classifier model with specified hyperparameters and trains it on the feature matrix X and target vector y using the classify() function. The hyperparameters used are n_estimators, min_samples_split, max_depth, and max_features. The code is using the scikit-learn library in Python.

**Confusion Matrix:**
A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.

**Save The Model and Conclusion:**
To pass the new testing dataset into trained model, save the model with best accurate classifier,then all attributes in new dataset as like trained attributes,equal no. of columns and same datatype. then i converted dataframe into csv file ,finally i predicted 'is the application is eligible for loan or not'.


**GUI (Graphical User Interface):**
GUI (Graphical User Interface) based loan status prediction application using machine learning. It uses the Tkinter library for creating the GUI interface and joblib and pandas libraries for machine learning and data processing tasks.
The GUI interface contains 11 input fields labeled Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, and Property_Area, respectively, to take user input. The user has to enter values for these fields and then click on the "Predict" button to get the loan status prediction.
Once the user clicks on the "Predict" button, the show_entry() function is called, which extracts the user inputs from the input fields and converts them into float data type. Then it creates a pandas DataFrame object to store these inputs and then applies a pre-trained machine learning model (loaded using the joblib library) to make the loan status prediction.
The model predicts whether the loan is approved or not based on the user's input values and returns the result as either "Loan approved" or "Loan Not Approved". The result is then displayed on the GUI using the Label widget.

