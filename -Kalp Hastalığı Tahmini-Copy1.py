#!/usr/bin/env python
# coding: utf-8

# #### Kütüphanelerin Yüklenmesi

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm


# #### Veriyi Yükleme

# In[2]:


kalp_veri= pd.read_csv("kalp14buyuk.csv")
print("\nİlk 5 satır:")
kalp_veri.head()


# In[3]:


print("\nSon 5 satır:\n")
kalp_veri.tail()


# ##### Eksik Veri Kontrolü

# In[4]:


kalp_veri.isnull().sum()


# #### Veri Hakkında Bilgiler

# In[5]:


kalp_veri.shape


# In[6]:


kalp_veri.info()


# ##### İstatistiksel Bilgiler

# In[7]:


kalp_veri.describe()


# #### Lojistik Regresyon Varsayımlarının Kontrolü

# In[8]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_hesap(df):
    vif = pd.DataFrame()
    vif["variables"]= kalp_veri.columns
    vif["VIF"]= [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif)

vif_hesap(kalp_veri)


# In[9]:


kalp_veri_final= kalp_veri.drop(["kan basıncı", "max kalp hızı", "kolesterol", "talasemi", "yaş"], axis= 1)
kalp_veri_final


# In[10]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
# Use variance inflation factor to identify any significant multi-collinearity
def vif_hesap(df):
    vif = pd.DataFrame()
    vif["variables"]= kalp_veri_final.columns
    vif["VIF"]= [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return(vif)

vif_hesap(kalp_veri_final)


# In[11]:


sns.set_theme(style= "darkgrid")
st_dep_box= sns.boxplot(data= kalp_veri_final[["st depresyonu"]]).set_title("ST Depresyonu Box Plot")
plt.show()


# In[12]:


kalp_veri_final.loc[(kalp_veri_final["st depresyonu"]>4),"st depresyonu"] = 1.07


# In[13]:


sns.set_theme(style= "darkgrid")
st_dep_box= sns.boxplot(data= kalp_veri_final[["st depresyonu"]]).set_title("ST Depresyonu Box Plot")
plt.show()


# In[14]:


sns.set_theme(style= "darkgrid")
st_depresyon= sns.regplot(x= "st depresyonu", y= "sonuç", data= kalp_veri_final, logistic= True).set_title("ST Depresyonu Log Odds Linear Plot")
plt.show()


# #### Görselleştirme

# In[15]:


kalp_veri_final.hist(figsize= (14,14))
plt.show()


# In[16]:


plt.figure(figsize= (15,15))
h= sns.heatmap(kalp_veri_final.corr(), annot= True, cmap="terrain", fmt= ".0%", linewidths= 0.3)
plt.show()


# In[17]:


sns.set_style("darkgrid")
sns.countplot(x= "sonuç", data= kalp_veri_final, palette= "bright")
kalp_veri_final["sonuç"].value_counts()


# In[18]:


sns.countplot(x= "cinsiyet", data= kalp_veri_final, palette= "bright")


# In[19]:


pic1= kalp_veri[kalp_veri["sonuç"]==1]
pic0= kalp_veri[kalp_veri["sonuç"]==0]


r= plt.subplot2grid((1,2),(0,0))
sns.countplot(pic0["cinsiyet"], palette= "viridis")
plt.title("SAĞLIKLI KİŞİLERİN CİNSİYET DAĞILIMLARI", fontsize= 10, weight= "bold")

plt.show()


r= plt.subplot2grid((1,2),(0,1))
sns.countplot(pic1["cinsiyet"], palette= "viridis")
plt.title("HASTALARIN CİNSİYET DAĞILIMLARI", fontsize= 10, weight= "bold")

plt.show()



# #### Veriyi Bölme

# In[20]:


x= kalp_veri_final.iloc[:,0:8]
print("\nBağımlı Değişkenler\n")
print(x)
y= kalp_veri_final.iloc[:,8]
print("\nBağımsız Değişken\n")
print(y)


# #### Veriyi Eğitim ve Test Kümesine Ayırma

# In[21]:


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.33, stratify=y, random_state=0)
print(x_train.shape, x_test.shape, y_test.shape)


# ##### Lojistik Regresyon

# In[22]:


LR= LogisticRegression()
LR.fit(x_train, y_train)


# ###### Model Başarısı

# In[23]:


tahmin_lr = LR.predict(x_test)

basari_orani_lr= accuracy_score(tahmin_lr, y_test)
print("\nLojistik Regresyon Tahmin Doğruluğu:", basari_orani_lr)


# ###### Karmaşıklık Matrisi

# In[24]:


cm_lr= confusion_matrix(y_test,tahmin_lr)
print("\nLojistik Regresyon Confusion Matrix:\n")
print(cm_lr)
#Gerçekte sağlam olup sağlam tahmin edilen kişi sayısı 135
#Gerçekte sağlam olup hasta tahmin edilen kişi sayısı 30
#Gerçekte hasta olup sağlam tahmin edilen kişi sayısı 38
#Gerçetke hasta olup hasta tahmin edilen kişi sayısı 136


# ##### Sonuçlar

# In[25]:


log_reg = sm.Logit(y_train, x_train).fit()
print(log_reg.summary())


# In[26]:


#kan şekeri parametresi istatistiksel olarak anlamlı olmadığı için veri setimizden çıkarılmıştır.
kalp_veri_final= kalp_veri_final.drop(["kan şekeri"], axis= 1) 


# ##### Final Veri Setimiz

# In[27]:


kalp_veri_final


# #### Veriyi Bölme

# In[28]:


x= kalp_veri_final.iloc[:,0:7]
print("\nBağımlı Değişkenler\n")
print(x)
y= kalp_veri_final.iloc[:,7]
print("\nBağımsız Değişken\n")
print(y)


# #### Veriyi Eğitim ve Test Kümesine Ayırma

# In[29]:


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.33, stratify=y, random_state=0)
print(x_train.shape, x_test.shape, y_test.shape)


# #### Kullanılan Algoritmalar

# ##### 1- Lojistik Regresyon

# In[30]:


LR= LogisticRegression()
LR.fit(x_train, y_train)


# ###### Model Başarısı

# In[31]:


tahmin_lr= LR.predict(x_test)

basari_orani_lr= accuracy_score(tahmin_lr, y_test)
print("\nLojistik Regresyon Tahmin Doğruluğu:", basari_orani_lr)


# ###### Karmaşıklık Matrisi

# In[32]:


cm_lr= confusion_matrix(y_test,tahmin_lr)
print("\nLojistik Regresyon Confusion Matrix:\n")
print(cm_lr)
#Gerçekte sağlam olup sağlam tahmin edilen kişi sayısı 135
#Gerçekte sağlam olup hasta tahmin edilen kişi sayısı 30
#Gerçekte hasta olup sağlam tahmin edilen kişi sayısı 37
#Gerçetke hasta olup hasta tahmin edilen kişi sayısı 137


# ##### Sonuçlar

# In[33]:


log_reg= sm.Logit(y_train, x_train).fit()
print(log_reg.summary())


# ##### 2- K-Nearest Neighbors

# In[34]:


KNN= KNeighborsClassifier(n_neighbors= 5, metric= "minkowski")
KNN.fit(x_train,y_train)


# ##### Model Başarısı

# In[35]:


tahmin_knn= KNN.predict(x_test)
basari_orani_knn= accuracy_score(tahmin_knn, y_test)
print("\nK-Nearest Neighbor Tahmin Doğruluğu:", basari_orani_knn)


# ##### Karmaşıklık Matrisi

# In[36]:


cm_knn= confusion_matrix(y_test, tahmin_knn)
print("\nK-Nearest Neighbor Confusion Matrix:\n")
print(cm_knn)
#Gerçekte sağlam olup sağlam tahmin edilen kişi sayısı 134
#Gerçekte sağlam olup hasta tahmin edilen kişi sayısı 31
#Gerçekte hasta olup sağlam tahmin edilen kişi sayısı 21
#Gerçetke hasta olup hasta tahmin edilen kişi sayısı 153


# ##### 3- Naive Bayes

# In[37]:


NB=GaussianNB()
NB.fit(x_train, y_train)


# ##### Model Başarısı

# In[38]:


tahmin_nb= NB.predict(x_test)
basari_orani_nb=accuracy_score(tahmin_nb, y_test)
print("\nNaive Bayes Tahmin Doğruluğu:", basari_orani_nb)


# ##### Confusion Matrix

# In[39]:


cm_nb= confusion_matrix(y_test, tahmin_nb)
print("\nNaive Bayes Confusion Matrix:\n")
print(cm_nb)
#Gerçekte sağlam olup sağlam tahmin edilen kişi sayısı 128
#Gerçekte sağlam olup hasta tahmin edilen kişi sayısı 37
#Gerçekte hasta olup sağlam tahmin edilen kişi sayısı 31
#Gerçetke hasta olup hasta tahmin edilen kişi sayısı 143


# ##### 4- Decision Tree

# In[40]:


DT= DecisionTreeClassifier(criterion= "entropy")
DT.fit(x_train, y_train)


# #### Model Başarısı

# In[41]:


tahmin_dt= DT.predict(x_test)
basari_orani_dt=accuracy_score(tahmin_dt, y_test)
print("\nDecision Tree Tahmin Doğruluğu:", basari_orani_dt)


# ##### Karmaşıklık Matrisi

# In[42]:


cm_dt= confusion_matrix(y_test, tahmin_dt)
print("\nDecision Tree Confusion Matrix:\n")
print(cm_dt)
#Gerçekte sağlam olup sağlam tahmin edilen kişi sayısı 159
#Gerçekte sağlam olup hasta tahmin edilen kişi sayısı 6
#Gerçekte hasta olup sağlam tahmin edilen kişi sayısı 10
#Gerçetke hasta olup hasta tahmin edilen kişi sayısı 164


# ##### 5- Support Vector Machine

# In[43]:


SVM=SVC(kernel= "rbf")
SVM.fit(x_train, y_train)


# ##### Model Başarısı

# In[44]:


tahmin_svm= SVM.predict(x_test)
basari_orani_svm= accuracy_score(tahmin_svm, y_test)
print("\nSupport Vector Machine Tahmin Doğruluğu:", basari_orani_svm)


# ##### Karmaşıklık Matrisi

# In[45]:


cm_svm= confusion_matrix(y_test, tahmin_svm)
print("\nSupport Vector Machine Confusion Matrix:\n")
print(cm_svm)
#Gerçekte sağlam olup sağlam tahmin edilen kişi sayısı 141
#Gerçekte sağlam olup hasta tahmin edilen kişi sayısı 24
#Gerçekte hasta olup sağlam tahmin edilen kişi sayısı 11
#Gerçetke hasta olup hasta tahmin edilen kişi sayısı 163


# ##### 6- Random Forest

# In[46]:


RF= RandomForestClassifier(n_estimators= 10, criterion= "entropy")
RF.fit(x_train, y_train)


# ##### Model Başarısı

# In[47]:


tahmin_rf= RF.predict(x_test)
basari_orani_rf=accuracy_score(tahmin_rf, y_test)
print("\nRandom Forest Tahmin Doğruluğu:", basari_orani_rf)


# ##### Karmaşıklık Matrisi

# In[48]:


cm_rf= confusion_matrix(y_test, tahmin_rf)
print("\nRandom Forest Confusion Matrix:\n")
print(cm_rf)
#Gerçekte sağlam olup sağlam tahmin edilen kişi sayısı 158
#Gerçekte sağlam olup hasta tahmin edilen kişi sayısı 7
#Gerçekte hasta olup sağlam tahmin edilen kişi sayısı 12
#Gerçetke hasta olup hasta tahmin edilen kişi sayısı 162


# #### Tahmin Sistemi

# In[49]:


#inputveri=(60,1,2,140,185,0,0,155,0,3,1,0,2,0)
inputveri= (1,2,0,0,3,1,0)
inputveri= np.asarray(inputveri)

#inputveriyi numpy array'e dönüştürme
inputveri= inputveri.reshape(1,-1)
tahmin= LR.predict(inputveri)
print(tahmin)

if (tahmin[0]==0):
    print("Kişinin kalp hastalığı yoktur")
else:
    print("Kişinin kalp hastalığı vardır")


# In[ ]:




