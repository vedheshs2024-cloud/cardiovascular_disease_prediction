import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures


st.set_page_config(page_title = "Major Project")
st.header("Spotify Songs’ Genre Segmentation")

st.text("Jupyter NoteBook Link ")
st.link_button("Open Jupyter program", "https://drive.google.com/file/d/1Sd8sgctUG41YtGz4qSucUguM0v_MNA-b/view?usp=drivesdk")

df = pd.read_csv(r"D:\blender\cardio_train (1).csv", sep=';')



df['age'] = (df['age'] / 365).astype(int)

st.dataframe(df)

col=['height', 'weight', 'ap_hi', 'ap_lo']

fig = plt.figure(figsize=(12,10))
sns.boxplot(data=df,x="weight")
st.subheader("---BoxPlot---")
st.pyplot(fig)


st.write("✅ Remove outliers based on logical bounds")

df_cleaned = df[
    (df['height'] >= 100) & (df['height'] <= 220) &
    (df['weight'] >= 50) & (df['weight'] <= 180) &
    (df['ap_hi'] > 0) & (df['ap_hi'] < 250) &
    (df['ap_lo'] > 0) & (df['ap_lo'] < 200) &
    (df['ap_hi'] > df['ap_lo']) &
    (df['gender'].isin([1, 2])) &
    (df['cholesterol'].isin([1, 2, 3])) &
    (df['gluc'].isin([1, 2, 3])) &
    (df['smoke'].isin([0, 1])) &
    (df['alco'].isin([0, 1])) &
    (df['active'].isin([0, 1])) &
    (df['cardio'].isin([0, 1]))
]
df=df_cleaned.copy()
df=df.copy(deep=True)

for i in col:
    q25,q75=np.percentile(df[i],25),np.percentile(df[i],75)
    IQR = q75 - q25
    Threshold=IQR*1.5
    lower,upper=q25-Threshold,q75+Threshold
    outliers=[j for j in df[i] if j < lower or j > upper]
    for k in outliers:
        data_1=df.drop(df.index[df[i]==k],axis=0)

df=data_1.copy()

st.latex("countplot for Traget variable")

fg = plt.figure(figsize=(12,10))
sns.countplot(x='cardio', data=df)
plt.title("cardioascular disease distribution")
st.pyplot(fg)

x = df.drop('cardio', axis=1)
y = df['cardio']
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)

fog  = plt.figure(figsize=(12,10))
sns.histplot(df['age'], kde=True)
plt.title("Age distribution (in years)")
plt.xlabel("Age (years)")
st.pyplot(fog)


corr = df.corr()
fig  = plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("correlation Matrix")
st.pyplot(fig)

x_train,x_test,y_train,y_test = train_test_split(x_scaled, y, test_size=0.4, random_state=40)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

model_1 = SVC()
model_1.fit(x_train, y_train)

col1,col2,col3,col4 = st.columns(4)

with col1:
    idi = st.number_input("ID:")
with col2:
    age = st.number_input("AGE:")
with col3:
    gender = st.number_input("Gender(1:Male,2:Female):")
with col4:
    height = st.number_input("Height:")

col5,col6,col7,col8 = st.columns(4)

with col5:
    weight = st.number_input("Weight:")
with col6:
    ap_hi = st.number_input("Ap_hi:")
with col7:
    ap_lo = st.number_input("Ap_lo:")
with col8:
    cholesterol = st.number_input("Cholesterol(No:0,Yes:1):")

col9,col10,col11,col12 = st.columns(4)

with col9:
    gluc = st.number_input("Glucose(No:0,Yes:1):")
with col10:
    smoke = st.number_input("Smoke(No:0,Yes:1):")
with col11:
    alco = st.number_input("Alcohol(No:0,Yes:1):")
with col12:
    active = st.number_input("Active(No:0,Yes:1):")
 
def prediction(idi,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active):
    value = [[idi,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active]]
    value_1 = pd.DataFrame(value)
    
    x_value = scalar.fit_transform(value_1)
    y_value = model_1.predict(x_value)
    st.write("cardiovascular_disease_presiction:")
    if y_value == 0:
        st.write("Be Happy you don't have cardiovascular_disease")
    if y_value == 1:
        st.write("Sorry! you have cardiovascular_disease")

prediction(idi,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active)
