import numpy as np
import pandas as pd
import warnings
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer




df = pd.read_csv("insurance.csv")
df_cleaned = df.copy()

cat_col = ["sex","smoker"]
encoder = OrdinalEncoder()
df_cleaned[cat_col] = encoder.fit_transform(df_cleaned[cat_col])

df_cleaned.drop_duplicates(inplace=True)
cat_col = ["region"]
onehotencoder = OneHotEncoder(sparse_output=False)
encoded =  onehotencoder.fit_transform(df_cleaned[cat_col])

encoded_df = pd.DataFrame(encoded,columns = ['northeast', 'northwest', 'southeast', 'southwest'], index = df_cleaned.index)
df_cleaned.drop(columns = ["region"],inplace = True)
df_cleaned = pd.concat([df_cleaned,encoded_df] , axis = 1)
df_cleaned["bmi_category"] = pd.cut(df_cleaned["bmi"],bins = [0.0,18.5,25.0,29.9,np.inf], labels =["underweight","nomralweight","overweight","obesity"])

encoded_df = onehotencoder.fit_transform(df_cleaned[["bmi_category"]])
encoded_df = pd.DataFrame(encoded_df,columns = ['nomralweight', 'obesity', 'overweight', 'underweight'],index = df_cleaned.index)
df_cleaned = pd.concat([df_cleaned,encoded_df],axis= 1)

col = ["age","bmi","children"]
scalar =  StandardScaler()
df_cleaned[col] = scalar.fit_transform(df_cleaned[col])

df_selected = [col for col in df_cleaned.columns if col != "charges" and col != "bmi_category" ]

cols = ["age","sex", "smoker", "bmi", "obesity","children", "southeast", "charges"]

final_df = df_cleaned[cols]

X = final_df.drop(columns = "charges",axis = 1)
Y = final_df["charges"]


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,shuffle= True,test_size=0.2 )

RandomForest_model3 = RandomForestRegressor()
RandomForest_model3.fit(X_train,Y_train)
RandomForest_pred = RandomForest_model3.predict(X_test)
mse = mean_squared_error(Y_test, RandomForest_pred)
rmse = np.sqrt(mse)



RandomForest_crossval = -cross_val_score(RandomForest_model3, X, Y, cv=10, scoring="neg_mean_squared_error")
RandomForest_rmse = np.sqrt(RandomForest_crossval)

r2 = r2_score(Y_test,RandomForest_pred)

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)




print("R2 accuray is and adjusted value is  :", r2 , adjusted_r2 )




print("Random Forest RMSE mean:", RandomForest_rmse.mean())

print("RandomForest_pred_rms ",rmse)





