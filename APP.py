# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import joblib
#from keras.models import load_model
import emoji
import warnings
warnings.filterwarnings("ignore")

def main():  
    st.title("Credit Card Fraud Detection")
    header = st.empty()
    with header.container():
        st.image(image="ezgif-3-fe301cad75.jpg")
        st.write("It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.")
        st.write("In this project we have used a dataset of transactions made by credit cards in September 2013 by European cardholders\n \
            We have employed 4 different techniques to enable us predict fraudulent transactions.\n \
                Because the dataset is mostly made up of features that have been transformed because of confidentiality,\n \
                in this demo we would perform some predictions on random samples from the same dataset.")
        st.write("So go ahead and select a model to get started")

    # Dataset
    dataset = pd.read_csv("dataset.csv")
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

    def choose(modelname="",file=""):
        # Load the model
        model = joblib.load(file)

        header.empty()
        st.empty()   
        st.subheader(modelname)
        
        number = st.number_input(label="How many predictions would you like to see?",value=0,min_value=0, max_value=10)
        predict_type = st.radio("How would you want data to be selected?",
                                ('Random', 'Fraud-only', 'Genuine-only'), index=0)

        def fetchdata(number, type):
            if isinstance(number, str) or number == 0 or number > 10:
                pass
            else:
                if type == "random":
                    data = dataset.sample(n = number)
                    return data
                elif type == "fraud":
                    fruaddata = dataset.loc[dataset["Class"] == 1]
                    return fruaddata.sample(n = number)
                elif type == "genuine":
                    genuinedata = dataset.loc[dataset["Class"] == 0]
                    return genuinedata.sample(n = number)
                
        if st.button(label="Fetch data and predict"):
            if predict_type == 'Random':
                data = fetchdata(number, "random")
                st.dataframe(data) 
                
            elif predict_type == 'Fraud-only':
                data = fetchdata(number, "fraud")
                st.dataframe(data)

            elif predict_type == 'Genuine-only':
                data = fetchdata(number, "genuine")
                st.dataframe(data)

            # For prediction
            X = data.drop(['Class'], axis=1)
            predicted = model.predict(X)
            predicted = predicted.ravel().tolist()

            for i, value in enumerate(predicted):
                predicted = ""
                actual = ""
                if modelname == "Copula-Based Outlier Detection (COPOD)":
                    if value == 0:
                        predicted = "Genuine"
                    elif value == 1:
                        predicted = "Fraud"
                else:
                    if value == 1:
                        predicted = "Genuine"
                    elif value == -1:
                        predicted = "Fraud"

                if data.iloc[i]['Class'] == 1:
                    actual = "Fraud"
                elif data.iloc[i]['Class'] == 0:
                    actual = "Genuine"

                if (predicted == "Genuine" and actual == "Genuine") or (predicted == "Fraud" and actual == "Fraud"):
                    show = emoji.emojize('Our model did good :white_check_mark:')
                else:
                    show = emoji.emojize('Our model did bad :x:')
                
                st.write("Row"+str(i+1)+":  ", "Our model predicted a {}, and it is a {}. {}".format(predicted, actual, show))

    menu = ["<Select One>","Isolation Forest", "Local Outlier Factor", "COPOD"]  
    choice = st.sidebar.selectbox("Select a Model",menu)
    
    if choice == "<Select One>":
        pass
    if choice == "Isolation Forest":
        choose(modelname="Isolation Forest",file="ISF.pkl")
    if choice == "Local Outlier Factor":    
        choose(modelname="Local Outlier Factor",file="LOF.pkl")
    if choice == "COPOD":
        try:   
            choose(modelname="Copula-Based Outlier Detection (COPOD)",file="COPOD.pkl")
        except FileNotFoundError:
            header.empty()
            st.empty()   
            st.subheader("Copula-Based Outlier Detection (COPOD)")
            st.write("Coming soon...")
             
if __name__ == '__main__':    
    main()