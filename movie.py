import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

df=pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\movie dataset\\IMDb Movies India.csv", encoding='latin1')
df.info()
df.isnull().sum()
df.head(10)

#Data Cleaning
df.dropna(subset=df.columns[1:9],how='all',inplace=True)
df.dropna(subset=['Name','Year'],how='all',inplace=True)
df.drop_duplicates(['Name','Year'],keep='first',inplace=True)
df.info()

df.dropna(subset=['Year'],inplace=True)
df['Year']=df['Year'].str.extract(r'([0-9].{0,3})',expand=False)
df['Duration']=df['Duration'].str.extract(r'([0-9]+)',expand=False)

def get_mode_with_default(x):
    mode_result = x.mode()
    if not mode_result.empty:
        return mode_result[0]
    else:
        return 'unknown'  

df['Actor 1']=df['Actor 1'].fillna(df.groupby('Year')['Actor 1'].transform(get_mode_with_default))
df['Actor 2']=df['Actor 2'].fillna(df.groupby('Year')['Actor 2'].transform(get_mode_with_default))
df['Actor 3']=df['Actor 3'].fillna(df.groupby('Year')['Actor 3'].transform(get_mode_with_default))

df['Director']=df.groupby(['Year','Actor 1','Actor 2','Actor 3'])['Director'].transform(get_mode_with_default)

df['Duration']=pd.to_numeric(df['Duration'])

def get_mean_with_default(x):
    mean_result = x.mean()
    if not math.isnan(mean_result):        
            return round(mean_result)
    else:
        return 0
df['Duration']=df.groupby(['Year','Director','Actor 1','Actor 2','Actor 3'])['Duration'].transform(get_mean_with_default)
df['Rating']=df.groupby(['Director','Actor 1'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby(['Director','Actor 2'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby(['Director','Actor 3'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby(['Year','Director'])['Rating'].transform(lambda x:x.mean())
df['Rating']=df.groupby('Year')['Rating'].transform(lambda x:x.mean())
df['Year']=pd.to_numeric(df['Year'])

df['Votes']=df['Votes'].str.extract(r'([0-9]+)',expand=False)
df['Votes']=pd.to_numeric(df['Votes'])

df['Votes']=df.groupby(['Year','Rating'])['Votes'].transform(lambda x:x.mean())

df['Votes']=df.groupby('Year')['Votes'].transform(lambda x:x.mean())
df.info()

#EDA
#Year with best rating
rating_sum=df.groupby('Year')['Rating'].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(x='Year',y='Rating',data=rating_sum)
sns.scatterplot(x='Year',y='Rating',data=rating_sum,color='r')
plt.yticks(np.arange(0,3000,400))
plt.xticks(np.arange(1920,2025,5))
plt.ylabel('Ratings')
plt.xlabel('Years')
plt.title('Ratings Per Years')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

#Year with best average rating
rating_avg=df.groupby('Year')['Rating'].mean().reset_index()

plt.figure(figsize=(20,6))
sns.lineplot(x='Year',y='Rating',data=rating_avg)
sns.scatterplot(x='Year',y='Rating',data=rating_avg,color='r')
plt.yticks(np.arange(4,8,0.5))
plt.xticks(np.arange(1920,2025,5))
plt.ylabel('Average Ratings')
plt.xlabel('Years')
plt.title('Average Ratings Per Years')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

#Top 20 Directors by Frequency of Movies
top_20=df.groupby('Director')['Name'].count()[0:20]

sns.barplot(x=top_20.index,y=top_20.values,data=df,palette='viridis')
plt.xticks(rotation=90)
plt.ylabel('Frequency Of Movies')
plt.xlabel('Director')
plt.show()

#Does length of movie have any impact with the rating
corr_leng_rat=df['Duration'].corr(df['Rating'])
print(f"Correlation Of Duration And Rating is {corr_leng_rat}")
#show there is no impact of duration on rating

plt.figure(figsize=(8,6))
sns.scatterplot(x='Duration',y='Rating',data=df)
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.title('Duration Vs Rating')
plt.yticks(np.arange(4,8,0.5))
plt.show()

#Top 10 movies according to rating per year and overall.
overall=df.nlargest(10,'Rating')
overall=overall.reset_index(drop=True)
print("Top 10 Movies Overall:")
overall

top_10_per_year = pd.DataFrame()
for year in df['Year'].unique():
    year_df = df[df['Year'] == year]
    top_10_year = year_df.nlargest(10, 'Rating').sort_values(by='Rating', ascending=False)
    top_10_per_year = top_10_per_year.append(top_10_year)
    

top_10_per_year = top_10_per_year.reset_index(drop=True)   
print("\nTop 10 Movies Per Year:")
top_10_year

#Number of popular movies released each year.
rat_bool=df['Rating']>=6
vot_bool=df['Votes']>110
pop_df=df[vot_bool & rat_bool]
pop_df

#ML
df.dropna(inplace=True)
df.isnull().sum()

#df.reset_index()
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,6))

sns.boxplot(data=df,y='Rating',ax=ax[0][0])
ax[0][0].set_title('Ratings')
ax[0][0].set_xlabel('Ratings')

sns.boxplot(data=df,y='Duration',ax=ax[0][1])
ax[0][1].set_title('Duration')
ax[0][1].set_xlabel('Duration')

sns.boxplot(data=df,y='Votes',ax=ax[1][0])
ax[1][0].set_title('Votes')
ax[1][0].set_xlabel('Votes')

sns.boxplot(data=df,y='Year',ax=ax[1][1])
ax[1][1].set_title('Years')
ax[1][1].set_xlabel('Years')

plt.tight_layout()

plt.show()

def out(df,col,dis):
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    lower=q1-(iqr*dis)
    upper=q3+(iqr*dis)
    return lower,upper
votes_low,votes_up=out(df,'Votes',1.5)
vote_out_count=(df['Votes'] > votes_up) | (df['Votes'] < votes_low)
df['Votes'][vote_out_count].count()
df=df[(df['Votes']>votes_low) & (df['Votes']<votes_up)]
year_low,year_upper=out(df,'Year',1.5)
year_out_count=(df['Year']>year_upper) | (df['Year']<year_low)
df['Year'][year_out_count].count()
sns.heatmap(df.corr(),cmap='YlGnBu',annot=True)

#applying ML
from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
df['Name']=LB.fit_transform(df['Name'])
df['Genre']=LB.fit_transform(df['Genre'])
df['Director']=LB.fit_transform(df['Director'])
df['Actor 1']=LB.fit_transform(df['Actor 1'])
df['Actor 2']=LB.fit_transform(df['Actor 2'])
df['Actor 3']=LB.fit_transform(df['Actor 3'])
from sklearn.linear_model import LinearRegression
LR=LinearRegression()

from sklearn.model_selection import train_test_split
x=df.drop('Rating',axis=1)
y=df['Rating']

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=42)

LR.fit(train_x,train_y)
pre_test=LR.predict(test_x)
pre_test

pre_train=LR.predict(train_x)

from sklearn.metrics import r2_score
score_test=r2_score(test_y,pre_test)
score_train=r2_score(train_y,pre_train)
print("print r2_score",score_test)
print('print r2_score',score_train)

from sklearn.linear_model import Ridge
RL=Ridge(alpha=10.0)
RL.fit(train_x,train_y)
RL_pre_test=RL.predict(test_x)
RL_pre_train=RL.predict(train_x)
r2_RL_test=r2_score(test_y,RL_pre_test)
r2_RL_train=r2_score(train_y,RL_pre_train)
print("print r2_score",r2_RL_test)
print('print r2_score',r2_RL_train)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_y, RL_pre_test)
print(mse)

from sklearn.model_selection import GridSearchCV

param={'alpha':[0.01, 0.1, 1.0, 10.0]}
grid=GridSearchCV(estimator=RL,param_grid=param,cv=5)
grid.fit(train_x,train_y)

print(grid.best_params_,grid.best_estimator_)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf=KFold(n_splits=10,random_state=42,shuffle=True)
cv=cross_val_score(RL,x,y,cv=kf,n_jobs=-1)
print('Accuracy : ',cv.mean()*100)

from sklearn.model_selection import RepeatedKFold
rfk=RepeatedKFold(n_splits=10,random_state=42,n_repeats=5)
cv1=cross_val_score(RL,x,y,cv=rfk,n_jobs=-1)
print('Accuracy : ',cv1.mean()*100)

from lightgbm import LGBMRegressor
LGBMR = LGBMRegressor(n_estimators=100, random_state=60)
LGBMR.fit(train_x, train_y)
lgbm_pre_test = LGBMR.predict(test_x)
lgbm_pre_train=LGBMR.predict(train_x)
r2_test_lgbm=r2_score(test_y,lgbm_pre_test)
r2_train_lgbm=r2_score(train_y,lgbm_pre_train)
print("print r2_score",r2_test_lgbm)
print('print r2_score',r2_train_lgbm)
mse_lgbm = mean_squared_error(test_y, RL_pre_test)
print(mse_lgbm)

cv2=cross_val_score(LGBMR,x,y,cv=kf,n_jobs=-1)
print('Accuracy : ',cv2.mean()*100)

cv3=cross_val_score(LGBMR,x,y,cv=rfk,n_jobs=-1)
print('Accuracy : ',cv3.mean()*100)
