#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('/Users/kuldeeppatel/Downloads/adult.csv')
data.head(10)
# %%
print(data.shape)
# %%
data.isin(['?']).sum()
# %%
data['workclass'] = data['workclass'].replace('?', np.nan)
data['occupation'] = data['occupation'].replace('?', np.nan)
data['native-country'] = data['native-country'].replace('?', np.nan)
# %%
#Check for missing value 
info = pd.DataFrame(data.isnull().sum(),columns=["IsNull"])
info.insert(1,"IsNa",data.isna().sum(),True)
info.insert(2,"Duplicate",data.duplicated().sum(),True)
info.insert(3,"Unique",data.nunique(),True)
info.insert(4,"Min",data.select_dtypes(include='number').min(),True)
info.insert(5,"Max",data.select_dtypes(include='number').max(),True)
info.T


# %%
df = data.copy()
# %%
df.dropna(how='any', inplace=True)
df
# %%
df.shape
# %%
df = df.drop_duplicates()
# %%

# %%
df.info()


# %%
df['education'].unique()
# %%
df['educational-num'].unique()
# %%
#'education and educational columnn both conveying same this one in str and one in numeric so drop educational-num

#[ 'capital-gain' ] & [ 'capital-loss' ] both columns have 75% data as 0.00 So, we can drop [ 'capital-gain' ] & [ 'capital-loss' ] both columns

df1 = df.drop(['educational-num','capital-gain','capital-loss'], axis=1)
# %%
df1.info()
# %%

from sklearn import preprocessing
# %%
label_encoder = preprocessing.LabelEncoder()

df1["gender"] = label_encoder.fit_transform(df1['gender'])
df1["workclass"] = label_encoder.fit_transform(df1['workclass'])
df1["education"] = label_encoder.fit_transform(df1['education'])
df1["marital-status"] = label_encoder.fit_transform(df1['marital-status'])
df1['occupation'] = label_encoder.fit_transform(df1['occupation'])
df1['relationship'] = label_encoder.fit_transform(df1['relationship'])
df1['race'] = label_encoder.fit_transform(df1['race'])
df1['native-country'] = label_encoder.fit_transform(df1['native-country'])
df1['income'] = label_encoder.fit_transform(df1['income'])

# %%
df
# %%
info = pd.DataFrame(df1.isnull().sum(),columns=["IsNull"])
info.insert(1,"IsNa",df1.isna().sum(),True)
info.insert(2,"Duplicate",df1.duplicated().sum(),True)
info.insert(3,"Unique",df1.nunique(),True)
info.insert(4,"Min",df1.min(),True)
info.insert(5,"Max",df1.max(),True)
info.T  
# %%
#Correlation matrix
import seaborn as sns
f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df1.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
# %%


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

X = df1.drop(columns={"income"}, axis=1)
y = df1["income"].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# %%
from sklearn.preprocessing import MinMaxScaler
sk = MinMaxScaler()
X_train = sk.fit_transform(X_train)
X_test = sk.transform(X_test)

#Find Best K 
K = 20
error =[]
accuracy=[]
for i in range(1,K+1):
    knn= KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train,y_train)
    y_pred =knn.predict(X_test)
    error.append(1-metrics.accuracy_score(y_test,y_pred))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))


# %%
print("Accuracy :" ,metrics.accuracy_score(y_test,y_pred))
# %%
plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
plt.plot(range(1,21),error,'r-',marker='o')
plt.xlabel('Values of K')
plt.ylabel('Error')
plt.grid()
plt.title('Error vs K')

plt.subplot(1, 2, 2)
plt.plot(range(1,21),accuracy,'r-',marker='o')
plt.xlabel('Values of K')
plt.ylabel('accuracy')
plt.grid()
plt.title('accuracy vs K')
# %%
