from flask import Flask, render_template, request, jsonify

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
# Undersampling the normal transactions data
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)

#transactions_data = pd.read_csv('app\PS_20174392719_1491204439457_log.csv')

transactions_data = pd.read_csv('https://transaccionesbancarias.blob.core.windows.net/databanca/datatransaccionesbancariasbackup.csv')

# Removing the null values
transactions_data = transactions_data.dropna()

# Checking for null values
print(transactions_data.isnull().sum())

# Shows the shape of the dataset
print("(No. of rows, No. of columns) -->", transactions_data.shape)

# Runs the first five rows of the dataset
transactions_data.head()

# Check the distribution of data
print (transactions_data['isFraud'].value_counts(),'\n')
print(pd.value_counts(transactions_data.isFraud, normalize=True))

X = transactions_data.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis = 1)
Y = transactions_data.isFraud

# Reducing the majority class equal to minority class 
rus = RandomUnderSampler(sampling_strategy=1)

X_res, Y_res = rus.fit_resample(X, Y)

print('gaaaaaaaaaaaa')
print(X_res.shape, Y_res.shape)
print(pd.value_counts(Y_res))

# Defining the training, validation, & test split
def train_validation_test_split(
    X, Y, train_size=0.8, val_size=0.1, test_size=0.1, 
    random_state=None, shuffle=True):
  
    assert int(train_size + val_size + test_size + 1e-7) == 1
    
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val,    test_size=val_size/(train_size+val_size), 
        random_state=random_state, shuffle=shuffle)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

X_train, X_val, X_test, Y_train, Y_val, Y_test = train_validation_test_split(
X_res, Y_res, train_size=0.8, val_size=0.1, test_size=0.1, random_state=1)
model = LogisticRegression()
model.fit(X_train, Y_train)

#LogisticRegression()

Y_pred = model.predict(X_val)
var_accuracy_score=accuracy_score(Y_val, Y_pred)
var_roc_auc_score=roc_auc_score(Y_val, Y_pred)
print(classification_report(Y_val, Y_pred))
print('Validation Dataset:','\n')
print('Accuracy score: ', var_accuracy_score)
print('ROC AUC Score: ', var_roc_auc_score)
# Unpickle classifier¿

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/transaccionesbancariasatipicas', methods=['GET', 'POST'])
def transaccionesbancariasatipicas():
    # If a form is submitted
    
    if request.method == "GET":
        print('entra al post')

        prediction={'accuracyscore':"{0:.3%}".format(var_accuracy_score),'rocaucscore':"{0:.3%}".format(var_roc_auc_score)}
        
    else:
        prediction = ""
    print('variable de predicciones')
    print(prediction)
    #return render_template("transaccionesbancariasatipicas.html", output = prediction)
    return jsonify(prediction).headers.add('Access-Control-Allow-Origin', '*')

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
