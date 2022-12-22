# PREDICTING THE SALE PRICE OF BULLDOZERS USING MACHINE LEARNING
# In this program we are going to go through an example machine learning project with the goal of predicting the sale price of Bulldozers.
# THE DATA IS DOWNLOADED FROM THE KAGGLE BLUEBOOK FOR BULLDOZERS COMPETITIONS
''' THERE ARE 3 MAIN DATASETS
Train.csv: is the training set, which contains data through the end of 2011.
Valid.csv: is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout
the majority of the competition. Your score on this set is used to create the public leaderboard.
Test.csv: is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012.' \
Your score on the test set determines your final rank for the competition.'''
# Evaluation
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.
# For more on evaluation of this project check:
import time

'''https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview/evaluation'''
# NOTE: Our Goal is to minimize the RMSLE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
df=pd.read_csv("data/TrainAndValid.csv",parse_dates=["saledate"])
print(df.info())
print(df.isna().sum())
fig, ax= plt.subplots()
ax.scatter(df["saledate"][:1000],df["SalePrice"][:1000])
plt.show(block=True)

# SORT THE DATAFRAME BY SALEDATE
df=df.sort_values(by=["saledate"],ascending=True)
# print(df.saledate.head(20))
df_tmp=df.copy()
# MAKING A COPY OF THE DATAFRAME SO THAT IF WE DID SOMETHING WRONG WITH THE DATAFRAME WE STILL HAVE THE ORIGINAL ONE WITH US

# WE WILL ADD A DATETIME PARAMETER FOR SALEDATE COLUMN
df_tmp["saleYear"]=df_tmp.saledate.dt.year
df_tmp["saleMonth"]=df_tmp.saledate.dt.month
df_tmp["saleDay"]=df_tmp.saledate.dt.day
df_tmp["saleDayofWeek"]=df_tmp.saledate.dt.dayofweek
df_tmp["saleDayofYear"]=df_tmp.saledate.dt.dayofyear
# print(df_tmp.head().T)
df_tmp=df_tmp.drop("saledate",axis=1)
print(df_tmp.state.value_counts())
print(df_tmp.info())

# FIND THE COLUMN THAT CONTAIN STRING
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label]= content.astype("category").cat.as_ordered() # CONVERTING THEM TO CATEGORY TYPE #cat stand for category and as_ordered is to sort then alphabetically
print(df_tmp.info())

# THE THING WE DID ABOVE CHANGE THE DATATYPE OF ALL STRING INTO SOME NUMERICAL VALUES WE CAN CHECK THIS BY
print(df_tmp.state.cat.codes)

# WE NOW WILL WORK WITH THE MISSING VALUES
# Fill Numeric Missing values First
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)               # 1.auctioneerID 2.MachineHoursCurrentMeter
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # WE WILL ADD A BINARY COLUMN THAT WILL TELL THAT THE VALUE IS MISSING OR NOT
            df_tmp[label+"_is_missing"]=pd.isnull(content) # WE ARE JUST CREATING THIS COLUMN AS IT MIGHT BE NEEDED IN THE FUTURE
            # FILL THE MISSING VALUE WITH THE MEDIAN
            df_tmp[label]= content.fillna(content.median())
# WE FILLED UP THE NUMERIC VALUE BUT WE STILL HAVE SOME MISSING VALUE
print(df_tmp.isna().sum())

# FILLING AND TURNING CATEGORICAL DATA INTO NUMBERS
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)

# Turn Categorical variables into numbers and fill the missing values
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label+"_is_missing"]=pd.isnull(content)
        df_tmp[label]=pd.Categorical(content).codes+1   # WE JUST DID +1 AS CATEGORICAL DATATYPE ASSIGN -1 TO THE MISSING VALUES
print(df_tmp.isna().sum()) # NO MISSING VALUE AND ALL DATA IS NUMERIC

# MODELLING
# from sklearn.ensemble import RandomForestRegressor
# model=RandomForestRegressor(n_jobs=-1,random_state=42)
# model.fit(df_tmp.drop("SalePrice",axis=1), df_tmp["SalePrice"])
# print(time.time())

# SPLITTING THE DATA
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]
X_train, y_train=df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid =df_val.drop("SalePrice", axis=1), df_val.SalePrice
print(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape)

# AS PER THE EVALUATION WE NEED TO MINIMIZE THE RMSLE
from sklearn.metrics import mean_squared_log_error, mean_absolute_error,r2_score
def rmsle(y_test,y_preds):
     return np.sqrt(mean_squared_log_error(y_test,y_preds))
def show_score(model):
    train_preds =model.predict(X_train)
    val_preds=model.predict(X_valid)
    scores={"Training MAE": mean_absolute_error(y_train,train_preds),
            "Valid MAE": mean_absolute_error(y_valid,val_preds),
            "Traning RMSLE":rmsle(y_train,train_preds),
            "Valid RMSLE":rmsle(y_valid,val_preds),
            "Traning R^2":r2_score(y_train,train_preds),
            "Valid R^2":r2_score(y_valid,val_preds)}
    return scores
from sklearn.ensemble import RandomForestRegressor

# model=RandomForestRegressor(n_jobs=-1,random_state=42,max_samples=10000)   #max_sample is to cut down te time
# model.fit(X_train,y_train)
# print(show_score(model))
# # HYPER PARAMETER TUNING WITH RANDOMIZED SEARCHED CV
# from sklearn.model_selection import RandomizedSearchCV
# rf_grid={"n_estimators": np.arange(10,1000,10),
#          "max_depth": [None, 3, 5, 10],
#          "min_samples_split": np.arange(2, 20, 2),
#          "min_samples_leaf":  np.arange(1, 20, 2),
#          "max_features": [0.5, 1, "sqrt", "auto"],
#          "max_samples": [10000]}
# rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
#                                                     random_state=42),
#                               param_distributions=rf_grid,
#                               n_iter=5,
#                               cv=5,
#                               verbose=True)
# rs_model.fit(X_train,y_train)
# print(rs_model.best_params_)  # 'n_estimators': 130, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_samples': 10000, 'max_features': 'auto', 'max_depth': None
'''model1=RandomForestRegressor(n_estimators=130,
                             min_samples_split=2,
                             min_samples_leaf=5,
                             max_samples=10000,
                             max_features="auto",
                             max_depth=None
                             )
model1.fit(X_train,y_train)
print(show_score(model1))  '''
ideal_model=RandomForestRegressor(n_estimators=40,
                                  min_samples_leaf=1,
                                  min_samples_split=14,
                                  max_features=0.5,
                                  n_jobs=-1,
                                  max_samples=None,
                                  random_state=42)
# THIS FEATURES ARE FOUNDED BY SIR TAkING N_ITERS =100
ideal_model.fit(X_train,y_train)
print(show_score(ideal_model)) # A GREAT GREAT IMPROVEMENT CAN BE SEEN
# NOW WE WILL MAKE PREDICTION ON THE TEST DATA
df_test=pd.read_csv("data/Test.csv",low_memory=False,parse_dates=['saledate'])
print(df_test.head())
# test_preds=ideal_model.predict(df_test) WE CAN'T DIRECTLY PREDICT AS DATA TYPES ARE DIFFERNT AND SOME VALUES ARE MISSING TOO
def df_transform(df):
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayofWeek"] = df.saledate.dt.dayofweek
    df["saleDayofYear"] = df.saledate.dt.dayofyear
    df.drop("saledate", axis=1, inplace=True)
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # WE WILL ADD A BINARY COLUMN THAT WILL TELL THAT THE VALUE IS MISSING OR NOT
                df[label + "_is_missing"] = pd.isnull(content)  # WE ARE JUST CREATING THIS COLUMN AS IT MIGHT BE NEEDED IN THE FUTURE
                # FILL THE MISSING VALUE WITH THE MEDIAN
                df[label] = content.fillna(content.median())
        if not pd.api.types.is_numeric_dtype(content):
                df[label + "_is_missing"] = pd.isnull(content)
                df[label] = pd.Categorical(content).codes + 1
    return df
df_test=df_transform(df_test)
print(df_test.head())
print(df_test.isnull().sum())
# test_preds=ideal_model.predict(df_test) # WE CAN'T PREDICT AS COLUMN IN TRAIN SET IS 102 WHILE IN TEST SET IS 101
print(set(X_train.columns) - set(df_test.columns)) # {'auctioneerID_is_missing'}
df_test["auctioneerID_is_missing"]=False
test_preds= ideal_model.predict(df_test)
print(test_preds)
# All set now we only need it in the format kaggle demanded
df_preds=pd.DataFrame()
df_preds["Sales_ID"]=df_test["SalesID"]
df_preds["Sales_Prices"]=test_preds
print(df_preds)
df_preds.to_csv("data/test_predict.csv", index=False)