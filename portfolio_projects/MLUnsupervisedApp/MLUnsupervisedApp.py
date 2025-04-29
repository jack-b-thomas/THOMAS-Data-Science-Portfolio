# Jack Thomas 
# Portfolio Update 3 for MDSC 20009 
# 4/14/25

#importing relevant libraries.
import streamlit as st 
import seaborn as sns 
import pandas as pd


#importing relevant machine learning models from sklearn


#other important things that need to work
from streamlit_option_menu import option_menu
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#The following code is for the sidebar 
st.sidebar.subheader("Select Model")
with st.sidebar:
    model = option_menu(menu_title="", #page selection will be stored in "model" for later use, once the data and model have been chosen.  
                        options= ["Principle Component Analysis", #options defines each page (what machine learning models the app will offer).
                                  "K-means clustering"], 
                        icons=["file-spreadsheet-fill",
                               "bucket-fill",], #page (and title) icons are from: https://icons.getbootstrap.com/?q=tree
                        default_index=0, #orders the pages. 
                        orientation='vertical') #arranges menu vertically. 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#User can select from four pre-uploaded dataframe, using a selection box. 
st.sidebar.markdown("---")
st.sidebar.subheader("Select Your Dataset")
selection = st.sidebar.selectbox(label= "", 
                                 options= ["Iris", "Penguins", "Titanic", "NBA", "Other"]) #selection box with each dataframe.
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#an if statement that iterates over "selection" and reads the right dataframe. 
#New object called file_select stores the users choice.  
if selection == "Penguins":
    df = sns.load_dataset("penguins")
elif selection == "Iris": 
    df = sns.load_dataset("iris")
elif selection == "Titanic": 
    df = sns.load_dataset('titanic')
elif selection == "NBA": 
    df = pd.read_csv("Data/NBA Stats 202425 All Metrics  NBA Player Props Tool.csv")
#User can also upload a dataframe using a form and a submit button (only CSV files can be uploaded). 
#the users uploaded dataframe is stored in an object called file_upload.
#this is how the app determines whether to use file_select or file_upload for the df
elif selection == "Other":
    with st.sidebar.form("upload form",
                         clear_on_submit=True): #clear_on_submit allows for new submissions without reloading the page. 
        file_upload = st.file_uploader("Upload your own CSV file", 
                                   type = "csv",
                                   accept_multiple_files=False) #creates a form where the user can upload his or own csv file.
        st.form_submit_button(label="Submit", type = "primary") #submit button for the form
    if file_upload is None: 
        st.error("Please upload and submit a file!") #displays an error message that is more clear than the one that streamlit shows - tells the user to submit the file
    df = pd.read_csv(file_upload)#assigns the uploaded file the dataframe object the app will use for the models. 
#----------------------------------------------------------------------------------------------------------------------------------------------------
#following code is for the main webpage (not sideabar)
st.title("The Unsupervised Machine Learning Experience") #code for title.
st.markdown("Click [here](https://www.youtube.com/watch?v=nGrW-OR2uDk) to see a cool video about my favorite machine learning model!")
st.subheader("Data")
st.write(df.head())
st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#this section of code code is for machine learning models and visualization: 
#it is formatted in one big if statement based on the "model" seleciton from the sidebar based on what type of machine learning model is selected in the sidebar 
#this part of the if statement is for the linear regression model:
if model == "Linear Regression": #the following code is for the linear regression model: 
    st.subheader("If you uploaded your dataset:")
    st.markdown(" - Make sure your dataframe is formatted according to [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf). This is the best way to make sure that the model will work with your dataframe.")
    st.markdown(" - Convert any categorical variables into numeric variables (eg. 1 for true, 0 for false). Linear Regression only works with numeric type variables. This will help be more accurate, because it will have access to more features when making the predictions.")
    st.markdown(" - Account for any missing data. Linear Regression models cannot handle missing data. Thus, due to the many possible datasets this model might encounter, the app atoumatically drops observations with missing data.")
    st.markdown("---")