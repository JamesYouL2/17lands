import pygsheets
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.metrics import r2_score
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from lightgbm import LGBMRegressor
from bs4 import BeautifulSoup

c = pygsheets.authorize(service_file="./Test.json")
page = requests.get("https://www.17lands.com/color_ratings")

soup = BeautifulSoup(page.content, 'html.parser')


df_k=pd.read_excel("./Karsten.xlsx")
df_17l=pd.read_excel("./17lands.xlsx")
df_17l=df_17l.replace("-", 0.4)

datadf=df_17l.iloc[2::2]
datadf=datadf.drop('Name',axis=1)
carddf=df_17l.iloc[1::2]
carddf=carddf['Name']
datadf.index = datadf.index-1
df_17l=pd.concat([carddf,datadf], axis=1)

df_17l["Seen"]=df_17l.groupby('Rarity')['# Seen'].apply(lambda x: x / x.mean())
df_17l["Picked"]=df_17l.groupby('Rarity')['# Picked'].apply(lambda x: x / x.mean())
df_17l["Games"]=df_17l.groupby('Rarity')['# Games'].apply(lambda x: x / x.mean())
df_17l["SqrtGames"]=np.sqrt(df_17l["Games"])

pyplot.show(qqplot(df_17l["Seen"], line='s'))
pyplot.show(qqplot(np.sqrt(df_17l["Games"]), line='s'))

df_17l = pd.get_dummies(df_17l, prefix_sep="_", columns=['Rarity'])



#Merge DFs
df=pd.merge(df_k,df_17l,on="Name")
df.columns

#X=df[['Avg. Seen At', 'Avg. Taken At', 'Win Rate', 'Picked', 'Seen', 'Games', '# Seen', '# Picked', '# Games'
#,'Rarity_common','Rarity_uncommon','Rarity_rare','Rarity_mythic']]
X=df[['Seen', 'Games', 'Rarity_common', 'Rarity_uncommon', 'Rarity_mythic']]

y=df['Final Rating']

regressor = LGBMRegressor()  
regressor.fit(X,y) #training the algorithm

y_pred = regressor.predict(X)

print(r2_score(y,y_pred))

df['Predicted Rating'] = pd.DataFrame(y_pred)

#Google Spreadsheet
gc = pygsheets.authorize(service_file='./Test.json')

#open the google spreadsheet (where 'PY to Gsheet Test' is the name of my sheet)
sh = gc.open('ArenaDraft')
wks = sh[1]

wks.set_dataframe(df.sort_values('Predicted Rating', ascending=False),(1,1))

print(regressor.coef_,X.columns)

### Exploratory analysis
### NOT NECESSARY TO RUN MODEL
#no of features
nof_list=np.arange(1,10)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
cols=list(df_17l.columns)
cols.remove('Name')
cols.remove('Color')
X=df[cols]

for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
        print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

#Initializing RFE model
rfe = RFE(model, 6)             #Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  #Fitting the data to model
model.fit(X_rfe,y)
temp = pd.Series(rfe.support_, index=list(X.columns))
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues