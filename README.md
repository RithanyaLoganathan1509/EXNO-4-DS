# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
df = pd.read_csv("bmi.csv")  
print("Original Dataset:")
print(df.head())
```
<img width="328" height="169" alt="image" src="https://github.com/user-attachments/assets/f60ee0e8-cca5-41e8-94cb-6895493126ae" />

```
df = df.dropna()
df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print("\nStandard Scaled Data:")
print(df_std.head())
```
<img width="386" height="174" alt="image" src="https://github.com/user-attachments/assets/746da915-dc7e-49a1-89d1-79eecd519cf9" />

```
df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())
```
<img width="394" height="157" alt="image" src="https://github.com/user-attachments/assets/56287c55-4d4b-48e3-82ce-868e8c2f3b67" />

```
df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())
```
<img width="411" height="174" alt="image" src="https://github.com/user-attachments/assets/718896ba-f3d3-4295-b571-7f212133e9a2" />

```
df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())
print("\nFeature Scaling Completed Successfully.")
```
<img width="407" height="197" alt="image" src="https://github.com/user-attachments/assets/b698084e-fa31-4782-93cc-eb9da9a629b7" />

```
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df = pd.read_csv("income(1) (1).csv")
print("Dataset Preview:")
print(df.head())
```
<img width="829" height="463" alt="image" src="https://github.com/user-attachments/assets/ca1381b0-9dc0-4c19-8b53-8ac8291162de" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation',
                       'relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes
X = df.drop(columns=['SalStat'])
y = df['SalStat']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))

selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))

logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))
```
<img width="977" height="82" alt="image" src="https://github.com/user-attachments/assets/ff7a5868-6d24-48ee-b2b6-a2ea1440a7c4" />

<img width="900" height="38" alt="image" src="https://github.com/user-attachments/assets/bab880b9-5428-446f-b984-c7f46a81f0fe" />

```
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
embedded_selector = SelectFromModel(model, threshold='median')
embedded_selector.fit(X, y)
selected_features_embedded = X.columns[embedded_selector.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features_embedded], y,
    test_size=0.2,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy (Embedded Method):", accuracy)
```
<img width="1035" height="56" alt="image" src="https://github.com/user-attachments/assets/9c134628-48de-4c66-9d0e-38ee3c3598d5" />

# RESULT:
Thus the Feature Scaling and selection Executed successfully.
