#Kütüphane yüklemesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef

#verisetinin yüklenmesi

df=pd.read_csv('C:/Users/ARG/Desktop/term-deposit-marketing-2020.csv')
#Duration sütunu düşürüldü. 
df = df.drop(['duration'], axis =1)

print("Describe Dataset","\n")
print(df.info())
print(df.head(10))
print(df.describe())

#Sürekli Değişkenler
print("Continuous Variables","\n")
continous_vars=df.describe().columns
print(continous_vars)
#Kategorik değişkenler
print("Categorical Variables","\n")
categorical_vars=df.describe(include=[object]).columns
print(categorical_vars)

#Sürekli Değişkenlerin Görselleştirilmesi
print("Continuous Variables Visualization","\n")
df.hist(column=continous_vars,figsize=(16,16))
plt.show()

#Kategorik Değişkenlerin Görselleştirilmesi
print("Categorical Variables Visualization","\n")
fig, axes = plt.subplots(4, 3, figsize=(16, 16))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.3)
for i, ax in enumerate(axes.ravel()):
    if i > 8:
        ax.set_visible(False)
        continue
    sns.countplot(y = categorical_vars[i], data=df, ax=ax)
plt.show()

#Korelasyon Heat map
print("Correlation Matrix","\n")    
correlation=df.corr(method="pearson")
plt.figure(figsize=(25,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True)
plt.show()
#Sütunların birleştirilmesi
columns = df.select_dtypes(include=[object]).columns
df = pd.concat([df, pd.get_dummies(df[columns])], axis=1)
df = df.drop(['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month','y'], axis =1)
print("\n","Describe Dataset_v2","\n")
print(df.info(),"\n \n \n")

#SCALE
min_max_scaler=preprocessing.MinMaxScaler()
data_scaled=pd.DataFrame(min_max_scaler.fit_transform(df),columns=df.columns)

#Y'yi kategorik değişkenden ayırarak teste hazır hale getirme
y = data_scaled.y_yes
data_scaled = data_scaled.drop(['y_yes','y_no'], axis = 1)

#Train, Test
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.2, random_state=42)

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family=sm.families.Binomial())
logm1.fit().summary()

#Korealasyon Heat Map 2
plt.figure(figsize=(50,20))
sns.heatmap(data_scaled.corr(), annot= True) #Çok yüksek korelasyona sahip değişken yok.

#Model seçimi = Logistic Regression(KNN ve Decision Tree' de denenmiş. Başarı oranı düşük çıkmıştır.)
logreg = LogisticRegression()
rfe = RFE(logreg,20) #En önemli 20 değişkenin seçilmesi 
rfe = rfe.fit(data_scaled,y)
print(rfe.support_)
print(rfe.ranking_)
#Önemli olabilecek 20 feature ('age', 'balance', 'day', 'campaign',''job_housemaid','job_retired',''marital_married', ''education_tertiary'','housing_yes','loan_yes', 'contact_cellular' ,contact_unknown,'month_apr', 'month_aug', 'month_jan', 'month_jul', 'month_jun','month_mar','month_nov', 'month_oct' )
data_scaled.columns

col = ['age', 'balance', 'day', 'campaign','job_housemaid','job_retired','marital_married','education_tertiary','housing_yes','loan_yes','contact_cellular','contact_unknown','month_apr','month_aug', 'month_jan', 'month_jul', 'month_jun','month_mar','month_nov', 'month_oct' ]

#Feature bulunduktan sonra tekrar Logistic Regression 
logsk = LogisticRegression()
logsk.fit(X_train[col], y_train)


logm4 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family=sm.families.Binomial())
logm4.fit().summary()

#age features ının P değeri 0.05'den büyük. age'i çıkarıyorum. 
col = ['balance', 'day', 'campaign','job_housemaid','job_retired','marital_married','education_tertiary','housing_yes','loan_yes','contact_cellular','contact_unknown','month_apr','month_aug', 'month_jan', 'month_jul', 'month_jun','month_mar','month_nov', 'month_oct' ]


logm5 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family=sm.families.Binomial())
logm5.fit().summary()

#Şu anki değişkenlerle modeli tekardan oluşturuyorum. 
logsk = LogisticRegression()
logsk.fit(X_train[col], y_train)


#Tahmin işi
y_pred = logsk.predict(X_test[col])

#y_predi dataframe dönüştürelim. 
y_pred_df = pd.DataFrame(y_pred)
y_test_df = pd.DataFrame(y_test)

print("Confusion Matrix","\n")
score = round(accuracy_score(y_test, y_pred_df),3) 
cm1 = cm(y_test, y_pred_df)
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=.3, 
        square = True, cmap = 'PuBu')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score), size = 12)
plt.show()
print("\n")  
#Accuracy Score = %92.9


rfe=rfe.fit(X_train[col],y_train)
print("Feature Selection","\n")
print(X_train[col][X_train[col].columns[rfe.ranking_==1].values].columns,"\n")

#CROSS VALIDATION
accuracies = cross_val_score(estimator = logsk, X = X_train[col], y = y_train, cv = 5)
print("Accuracy (mean): %",accuracies.mean()*100)
print("std: %",accuracies.std()*100)
#Accuracy mean = %92.74
#Std = %0.08
scores = cross_val_score(logsk, X_train[col], y_train, scoring='neg_mean_absolute_error', cv=5)
print ("MAE (mean): %" , scores.mean())
#mae = %-0.07


mae = mean_absolute_error(logsk.predict(X_test[col]), y_test)
mse = mean_squared_error(logsk.predict(X_test[col]), y_test)
rmse = np.sqrt(mse)
print('Mean Absolute Error (MAE): %.2f' % mae) #0.07
print('Mean Squared Error (MSE): %.2f' % mse) #0.07
print('Root Mean Squared Error (RMSE): %.2f' % rmse) #0.27

print("Cohen Score",cohen_kappa_score(y_test, y_pred_df)) #0.05
print("Matthew Score",matthews_corrcoef(y_test, y_pred_df)) #0.12 #Sınıflandırma başarısı maalesef düşük.





