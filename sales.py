import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\sales data\\advertising.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()/df.shape[0]

fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
#line_kws parameter is used to set the line color to red, while the scatter_kws
sns.regplot(x='TV',y='Sales',data=df,ax=ax[0][0],line_kws={'color': 'red'}, scatter_kws={'color': 'blue'})
ax[0][0].set_xlabel('TV')
ax[0][0].set_ylabel('Sales')
ax[0][0].set_title('Sales Per TV Adds')

sns.regplot(x='Radio',y='Sales',data=df,ax=ax[0][1],line_kws={'color': 'red'}, scatter_kws={'color': 'blue'})
ax[0][1].set_xlabel('Radio')
ax[0][1].set_ylabel('Sales')
ax[0][1].set_title('Sales Per Radio Adds')

sns.regplot(x='Newspaper',y='Sales',data=df,ax=ax[1][0],line_kws={'color': 'red'}, scatter_kws={'color': 'blue'})
ax[1][0].set_xlabel('Newspaper')
ax[1][0].set_ylabel('Sales')
ax[1][0].set_title('Sales Per Newspaper Adds')

plt.show()

figsize=(10,8)

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')#annot show relation
plt.title('Relationship Between All Variable')
plt.show()


plt.figure(figsize=(12,8))

sns.lineplot(x='Sales',y='TV',data=df,color='r',label='TV')
sns.lineplot(x='Sales',y='Radio',data=df,color='blue',label='Radio')
sns.lineplot(x='Sales',y='Newspaper',data=df,color='black',label='Newspaper')
plt.xlabel('Sales')
plt.xticks(np.arange(0,33,3))
plt.ylabel('Platform')

plt.grid()
plt.show()


fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,8))

sns.histplot(df['TV'],ax=ax[0][0])
ax[0][0].set_xlabel('TV')
ax[0][0].set_ylabel('Frequency')
ax[0][0].set_title('TV')

sns.histplot(df['Newspaper'],ax=ax[0][1])
ax[0][1].set_xlabel('Newspaper')
ax[0][1].set_ylabel('Frequency')
ax[0][1].set_title('Newspaper')

sns.histplot(df['Radio'],ax=ax[1][0])
ax[1][0].set_xlabel('Radio')
ax[1][0].set_ylabel('Frequency')
ax[1][0].set_title('Radio')

sns.histplot(df['Sales'],ax=ax[1][1])
ax[1][1].set_xlabel('Sales')
ax[1][1].set_ylabel('Frequency')
ax[1][1].set_title('Sales')

plt.tight_layout()
plt.show()


fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,8))

sns.boxplot(df['Radio'],ax=ax[0][0])
ax[0][0].set_title('Radio')

sns.boxplot(df['Newspaper'],ax=ax[0][1])
ax[0][1].set_title('Newspaper')

sns.boxplot(df['TV'],ax=ax[1][0])
ax[1][0].set_title('Tv')

sns.boxplot(df['Sales'],ax=ax[1][1])
ax[1][1].set_title('Sales')

plt.show()


def quant(df,col,dis):
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    
    low=q1-(iqr*dis)
    upp=q3+(iqr*dis)
    return low,upp
lower,upper=quant(df,'Newspaper',1.5)
out=(df['Newspaper']<lower) | (df['Newspaper']>upper)
df['Newspaper'][out].count()

df=df[~out]
df.shape


#Applpying Ml
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
x=df.drop('Sales',axis=1)
y=df['Sales']
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.3,random_state=42)
LR=LinearRegression()
LR.fit(trainx,trainy)
pre_test=LR.predict(testx)
pre_train=LR.predict(trainx)
print('Accuarcy on Testing',r2_score(testy,pre_test))
print('Accuarcy on Training',r2_score(trainy,pre_train))


kf=F=KFold(n_splits=5,random_state=43,shuffle=True)
cv=cross_val_score(LR,x,y,cv=kf,n_jobs=-1)
print('Accuracy : ',cv.mean()*100)

rkf=RepeatedKFold(n_splits=5,n_repeats=15,random_state=5)
cv1=cross_val_score(LR,x,y,cv=rkf,n_jobs=-1)
print('Accuracy : ',cv1.mean()*100)

RL=Ridge()
RL.fit(trainx,trainy)
pre_test_r=RL.predict(testx)
pre_train_r=RL.predict(trainx)
print('Accuarcy on Testing',r2_score(testy,pre_test_r))
print('Accuarcy on Training',r2_score(trainy,pre_train_r))

kf=F=KFold(n_splits=5,random_state=43,shuffle=True)
cv3=cross_val_score(RL,x,y,cv=kf,n_jobs=-1)
print('Accuracy : ',cv3.mean()*100)

rkf=RepeatedKFold(n_splits=5,n_repeats=15,random_state=5)
cv4=cross_val_score(RL,x,y,cv=rkf,n_jobs=-1)
print('Accuracy : ',cv4.mean()*100)

