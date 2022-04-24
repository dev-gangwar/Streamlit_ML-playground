
from requests import request
import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("ML PlayGround")
st.write("""
# Explore different classifiers on different datasets:
Find which one is the best??
""")
dataset_name=st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))
classifier_name=st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))
def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Brest Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    x=data.data
    y=data.target
    return x,y
    
x,y=get_dataset(dataset_name)
st.write("Shape of dataset",x.shape)
st.write("Number of Classes",len(np.unique(y)))


def add_par_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K
    elif clf_name=="SVM":
        C=st.sidebar.slider("C",0.1,10.0)
        params["C"]=C
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        n_estimators=st.sidebar.slider("n_estimator",1,100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators
    return params

params=add_par_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name=="SVM":
        clf=SVC(C=params["C"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf

clf=get_classifier(classifier_name,params)

#classification
X_train,X_text,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_text)
acc=accuracy_score(Y_test,Y_pred)
st.write(f"classifier={classifier_name}")
st.write(f"accuracy={acc}")

#plotting
pca=PCA(2)
X_projected=pca.fit_transform(x)
x1=X_projected[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()
st.pyplot(fig)
