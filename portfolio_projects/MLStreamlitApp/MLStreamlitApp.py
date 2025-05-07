# Jack Thomas 
# Portfolio Update 3 for MDSC 20009 
# 4/14/25


#importing relevant libraries.
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import graphviz

#importing relevant machine learning models from sklearn
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#other important things that need to work
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


#following code is for the sidebar: 
#user will navigate data and machine learning models through a sidebar. 
# uses st.option_menu: https://discuss.streamlit.io/t/streamlit-option-menu-is-a-simple-streamlit-component-that-allows-users-to-select-a-single-item-from-a-list-of-options-in-a-menu/20514.
st.sidebar.subheader("Select Model")
with st.sidebar: 
    model = option_menu(menu_title = "", #page selection will be stored in "model" for later use, once the data and model have been chosen. 
                        options= ["Linear Regression", #options defines each page (what machine learning models the app will offer).
                                  "Logistic Regression", 
                                  "K-Nearest Neighbor",
                                  "Decision Tree"],
                        icons=["graph-up",
                               "bezier2",
                               "people-fill",
                               "diagram-2-fill"], #page (and title) icons are from: https://icons.getbootstrap.com/?q=tree.
                        default_index=0, #orders the pages.  
                        orientation='vertical') #arranges menu vertically. 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#User can select from four pre-uploaded dataframe, using a selection box. 
st.sidebar.markdown("---")
st.sidebar.subheader("Select Your Dataset")
selection = st.sidebar.selectbox(label= "", 
                                 options= ["Iris", "Penguins", "Titanic", "Other"]) #selection box with each dataframe. 



#-------------------------------------------------------------------------------------------------------------------------------------------------------
#an if statement that iterates over "selection" and reads the right dataframe. 
#New object called file_select stores the users choice.  
if selection == "Penguins":
    df = sns.load_dataset("penguins")
elif selection == "Iris": 
    df = sns.load_dataset("iris")
elif selection == "Titanic": 
    df = sns.load_dataset('titanic')
#----------------------------------------------------------------------------------------------------------------------------------------------------
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
st.title("The Machine Learning Experience") #code for title.
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
#----------------------------------------------------------------------------------------------------------------------------------------------------    
    #Here, we define the target and feature variables using the chosen dataset: 
    st.subheader("Defining target and feature variables")
    df_num = df.select_dtypes(include='number') #only numeric variables because of the requirement for lineaer regression. 
    drop_cols = st.multiselect(label="Drop any columns you do not be a part of the model as either features or targets (eg. if your dataset has a unique id label for each column it can be dropped here).", #the multi-selection box lets user drop any columns they do not want to be feature variables in the model. 
                             options= df_num.columns,
                             help="If your dataframe has any irrelevant columns, like a unique observation label, you can drop them by selecting it here.")
    df_rel = df_num.drop(df_num[drop_cols], axis=1)#allows the user to drop any irrelevant colomns
    df_clean = df_rel.dropna(axis = 0)#dropping any observations with NaNs
    target = st.selectbox(label="Choose a Variable to Predict:",  #the selection box lets users choose which variable they want to predict
                          options = df_clean.columns, 
                          help="Choose one of the numeric columns to be your target variable. This is what the model will be predicting. Be sure to input it exactly as it appears in the uploaded file.")
    y = pd.Series(df_clean[target])#defining the target variable that will be used in the model. 
    feature_vars = pd.DataFrame(df_clean)
    X = feature_vars.drop(target, axis=1)#defining the feature variables that will be used in the model. 
#----------------------------------------------------------------------------------------------------------------------------------------------------   
    #displaying both target and feature variables for the user.
    features_column, target_column = st.columns([1,1], gap="small", vertical_alignment= "top")
    with features_column:
        st.write("Feature Variables") 
        st.write(X.head())
    with target_column:
        st.write("Target Variable")
        st.write(y.head())
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #Here, we  train and run the model: 
    st.subheader("Making the model")
    train_size = st.slider(label = "Choose the size of the training set", #allows the user choose the size of training set.  
                                  min_value=0.01,
                                  max_value=0.99, 
                                  value = 0.8)
    scale_selection = st.selectbox(label="Choose from scaled or unscaled data", #allows the user to choose between scaled and unscaled data.
                                   options= ["Unscaled", "Scaled"])
    st.markdown(" - For Linear Regression, unscaled or scaled features does not affect the results (ie your accuracy will still be the same). What will change, however, is the interprerabilty of your coefficients. With scaled features you will be able to see which features had a greater influence on the model's predictions.")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #the rest of the linear regression is coded in an if statement, to account for the scaled and unscaled models.
    if scale_selection == "Unscaled": 
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, #splitting the dataframe into training and testing sets 
                                                                    train_size= train_size, 
                                                                    random_state=42)
        lin_reg_raw = LinearRegression()#initializing the unscaled model 
        lin_reg_raw.fit(X_train_raw, y_train)#training the unscaled model 
        y_pred_raw = lin_reg_raw.predict(X_test_raw)#predicting the feature variable using the testing set and the unscaled model 
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #these are three metrics used to evaluate the model 
        mse_raw = mean_squared_error(y_test, y_pred_raw)
        rmse_raw = root_mean_squared_error(y_test, y_pred_raw)
        r2_raw = r2_score(y_test, y_pred_raw)
#----------------------------------------------------------------------------------------------------------------------------------------------------            
        #presenting the unscaled model's metrics and coefficients to the user 
        st.markdown("---")
        metrics_col, coef_col = st.columns([1,1], #two streamlit columns to facilitate ease of viewing.  
                                           gap="small", 
                                           vertical_alignment="top")
        with metrics_col: 
            st.subheader("Metrics")
            st.markdown("**Mean Squarred Error** measures the average squared difference between the estiamted values and the true values. **Root Mean Squared Error** measures the absolute value of the difference between the estimated value and the true value. **R squared** measures the percent of the variance in the data accounted for by the model.")
            st.text(f"Mean Squared Error: {mse_raw:.2f}")
            st.text(f"Root Mean Squared Error: {rmse_raw:.2f}")
            st.text(f"R² Score: {r2_raw:.2f}")
        with coef_col:
            st.subheader("Model Coefficients")
            st.markdown("These statistics show how much the target variable would changed based on a unit change in the feature variable (all other features are held constant).")
            st.text(pd.Series(lin_reg_raw.coef_,
                index=X.columns))
            st.markdown("The intercept shows the target variable value if all feature variables are set to 0.")
            st.text("Intercept")
            st.text(pd.Series(lin_reg_raw.intercept_))
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #Creating a graph of the model's residuals to show the user. 
        st.markdown("---")
        lin_reg_resid_raw = LinearRegression()#creating a model using the entire data set to show residuals. 
        lin_reg_resid_raw.fit(X, y)
        y_pred_resid_raw = lin_reg_resid_raw.predict(X)
        residuals = y - y_pred_resid_raw #creating the residuals object
        fig, ax = plt.subplots() #initializing the scatterplot of residuals and predictions 
        sns.scatterplot(x = y_pred_resid_raw, 
                        y = residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(f"Predicted {target}")
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs. Predicted {target}")
        sns.set_style(style="darkgrid")
        st.pyplot(fig)#displaying the scatterplot in streamlit
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #doing the same for scaled data
    else: 
        scaler = StandardScaler() #initializing a standard scaler to scale data 
        features_scaled = scaler.fit_transform(X) #scaling the data 
        X_scaled = pd.DataFrame(features_scaled, columns=X.columns) #putting the scaled data in a df
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, train_size=train_size, random_state=42)#splitting the scaled data into training and test sets. 
        lin_reg_scaled = LinearRegression()#initializing scaled linear regression model.
        lin_reg_scaled.fit(X_train_scaled, y_train_scaled)#training the scaled linear model. 
        y_pred_scaled = lin_reg_scaled.predict(X_test_scaled)#making predictions using the scaled linear model. 
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #same metrics, but for the scaled data: 
        mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
        rmse_scaled = root_mean_squared_error(y_test_scaled, y_pred_scaled)
        r2_scaled = r2_score(y_test_scaled, y_pred_scaled)
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #displaying the metrics and coefficients using two streamlit columns.  
        st.markdown("---")
        metrics_col, coef_col = st.columns([1,1], 
                                           gap="small", 
                                           vertical_alignment="top")
        with metrics_col: 
            st.subheader("Metrics")
            st.markdown("**Mean Squarred Error** measures the average squared difference between the estiamted values and the true values. **Root Mean Squared Error** measures the absolute value of the difference between the estimated value and the true value. **R squared** measures the percent of the variance in the data accounted for by the model.")
            st.text(f"Mean Squared Error: {mse_scaled:.2f}")
            st.text(f"Root Squared Error: {rmse_scaled:.2f}")
            st.text(f"R² Score: {r2_scaled:.2f}")
        with coef_col:
            st.subheader("Model Coefficients")
            st.markdown("These statistics show how much the target variable would changed based on a unit change in the feature variable (all other features are held constant).")
            st.text(pd.Series(lin_reg_scaled.coef_,
                index=X.columns))
            st.markdown("The intercept shows the target variable value if all feature variables are set to 0.")
            st.text("Intercept")
            st.text(pd.Series(lin_reg_scaled.intercept_))
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #making the same graph, but this time for the scaled residuals and predictions. 
        st.markdown("---")
        lin_reg_resid_scaled = LinearRegression()
        lin_reg_resid_scaled.fit(X_scaled, y)
        y_pred_resid_scaled = lin_reg_resid_scaled.predict(X_scaled)
        residuals_scaled = y - y_pred_resid_scaled
        fig, ax = plt.subplots()
        sns.scatterplot(x = y_pred_resid_scaled, 
                        y = residuals_scaled)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(f"Predicted {target}")
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs. Predicted {target}")
        sns.set_style(style="darkgrid")
        st.pyplot(fig)#plotting the figure in streamlit 
#----------------------------------------------------------------------------------------------------------------------------------------------------
#the following code is for the logistic regression machine learning model: 
elif model == "Logistic Regression": 
    st.subheader("If you uploaded your dataset:")
    st.markdown(" - Make sure your dataframe is formatted according to [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf). This is the best way to make sure that the model will work with your dataframe.")
    st.markdown(" - Convert any categorical variables into numeric variables (eg. 1 for true, 0 for false). Logistic Regression only works with numeric type variables. This will help be more accurate, because it will have access to more features when making the predictions.")
    st.markdown(" - Account for any missing data. Logistic Regression models cannot handle missing data. Thus, due to the many possible datasets this model might encounter, the app atoumatically drops observations with missing data.")
    st.markdown(" - Logistic Regression is pretty similar to Linear Regression, you'll find that the preprocessing steps look similar to those on the linear regression page.")
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #Here, we define the target and feature variables using the chosen dataset: 
    #This is code is the exact same as the code from the linear regression model (just splitting the data into training and testing sets) 
    st.subheader("Defining target and feature variables")
    df_num = df.select_dtypes(include='number') #only numeric variables because of the requirement for lineaer regression. 
    drop_cols = st.multiselect(label="Drop any columns you do not be a part of the model as either features or targets (eg. if your dataset has a unique id label for each column it can be dropped here).", #the multi-selection box lets user drop any columns they do not want to be feature variables in the model. 
                             options= df_num.columns,
                             help="If your dataframe has any irrelevant columns, like a unique observation label, you can drop them by selecting it here.")
    df_rel = df_num.drop(df_num[drop_cols], axis=1)#dropping NaN so model can run 
    df_clean = df_rel.dropna()
    target = st.selectbox(label="Choose a Variable to Predict. Since logistic regression is a classification model, your target variable (what you put here) must be a categorical variable, but encoded as numeric.",  #the selection box lets users choose which variable they want to predict
                          options = df_clean.columns, 
                          help="Choose one of the numeric columns to be your target variable. This is what the model will be predicting.")
    y = pd.Series(df_clean[target])#defining the target variable that will be used in the model. 
    feature_vars = pd.DataFrame(df_clean)
    X = feature_vars.drop(target, axis=1) #defining the feature variables that will be used in the model. 
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #displaying both target and feature variables for the user.
    features_column, target_column = st.columns([1,1], gap="small", vertical_alignment= "top")
    with features_column:
        st.write("Feature Variables") 
        st.write(X.head())
    with target_column:
        st.write("Target Variable")
        st.write(y.head())
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #Here the user can set a few hyperpermaters 
    st.subheader("Making the model")
    train_size = st.slider(label = "Choose the size of the training set", #allows the user choose the size of training set.  
                                  min_value=0.01,
                                  max_value=0.99, 
                                  value = 0.8)
    scale_selection = st.selectbox(label="Choose from scaled or unscaled data", #allows the user to choose between scaled and unscaled data.
                                   options= ["Unscaled", "Scaled"])
    st.markdown(" - For Logistic Regression, unscaled or scaled features does not affect the results (ie your accuracy will still be the same). What will change, however, is the interprerabilty of your coefficients. With scaled features you will be able to see which features had a greater influence on the model's predictions.")
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #The following code initiates, trains and runs the model: 
    if scale_selection == "Unscaled": 
         X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, #splitting the dataframe into training and testing sets 
                                                                    train_size= train_size, 
                                                                    random_state=42)
         log_reg_raw = LogisticRegression()
         log_reg_raw.fit(X_train_raw, y_train_raw)
         y_pred_raw = log_reg_raw.predict(X_test_raw)
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #These are the metrics the app will use to evaluate the logistic regresssion model:          
         accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)
         classification_report_raw = classification_report(y_test_raw, y_pred_raw, output_dict=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------
         #Displaying the evaluation metrics: 
         st.header("How did the model perform?")
         st.subheader("Accuracy Score")
         st.text(f"{accuracy_raw *100:.2f}% of predictions were correct.")#displaying accuracy
         coef_col, int_col = st.columns([1,1], # creating columns to display coefficients and intercepts
                                        vertical_alignment="top")
         with coef_col: 
             st.subheader("Model Coefficients")
             st.markdown("These statistics show how much the target variable would changed based on a unit change in the feature variable (all other features are held constant).")
             coef_raw = pd.Series(log_reg_raw.coef_[0], index=X.columns)
             st.text(coef_raw) # Display coefficients
         with int_col:
             st.subheader("Intercept")
             st.markdown("The intercept shows the target variable value if all feature variables are set to 0.")
             intercept_raw = log_reg_raw.intercept_
             st.text(intercept_raw) # Display intercept
         st.subheader("Classification Report")
         st.dataframe(classification_report_raw)# Display classification report
         st.markdown(" - **Precision** is the number of true positives divided the total number of predicted positives.")
         st.markdown(" - **Recall** is the number of true positives divided by the sum of true positives and false negatives.")
         st.markdown(" - The **f1-score** is the harmonic mean of the precision and recall statistics.")
         fig, ax = plt.subplots()#initializing the heatmap that will display the confusion matrix results.
         cm = confusion_matrix(y_test_raw, y_pred_raw)
         sns.heatmap(cm, 
                     annot=True,
                     fmt="d",
                     cmap="Reds")
         plt.xlabel(f"Predicted {target}")
         plt.ylabel(f"Actual {target}")
         plt.title("Confusion Matrix")
         st.pyplot(fig)#displaying the confusion matrix 
         st.markdown(" - The **Confusion Matrix** shows the ratios of true positives, false postives, true negatives, and false negatives.")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #The following code is for when the user chooses the data is scaled:
    #the code is largely the same as the code for the unscaled data 
    elif scale_selection == "Scaled":  
        scaler = StandardScaler() #initializing a standard scaler to scale data 
        features_scaled = scaler.fit_transform(X) #scaling the data 
        X_scaled = pd.DataFrame(features_scaled, columns=X.columns) #putting the scaled data in a df
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, #splitting the dataframe into training and testing sets 
                                                                    train_size= train_size, 
                                                                    random_state=42)
        log_reg_scaled = LogisticRegression()
        log_reg_scaled.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = log_reg_scaled.predict(X_test_scaled)
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #These are the metrics the app will use to evaluate the logistic regresssion model:          
        accuracy_scaled = accuracy_score(y_test_scaled, y_pred_scaled)
        classification_report_scaled = classification_report(y_test_scaled, 
                                                             y_pred_scaled, 
                                                             output_dict=True)
#----------------------------------------------------------------------------------------------------------------------------------------------------
#Displaying the evaluation metrics: 
        st.header("How did the model perform?")
        st.subheader("Accuracy Score")
        st.text(f"{accuracy_scaled * 100:.2f}% of predictions were correct.")
        coef_col, int_col = st.columns([1,1], # creating columns to display coefficients and intercepts
                                        vertical_alignment="top")
        with coef_col: 
             st.subheader("Model Coefficients")
             st.markdown("These statistics show how much the target variable would changed based on a unit change in the feature variable (all other features are held constant).")
             coef_scaled = pd.Series(log_reg_scaled.coef_[0], index=X_scaled.columns)
             st.text(coef_scaled) # Display coefficients
        with int_col:
             st.subheader("Intercept")
             st.markdown("The intercept shows the target variable value if all feature variables are set to 0.")
             intercept_scaled = log_reg_scaled.intercept_
             st.text(intercept_scaled) # Display intercept
        st.subheader("Classification Report")
        st.dataframe(classification_report_scaled)# Display classification report
        st.markdown(" - **Precision** is the number of true positives divided the total number of predicted positives.")
        st.markdown(" - **Recall** is the number of true positives divided by the sum of true positives and false negatives.")
        st.markdown(" - The **f1-score** is the harmonic mean of the precision and recall statistics.")
        fig, ax = plt.subplots()#initializing the heatmap that will display the confusion matrix results.
        cm = confusion_matrix(y_test_scaled, y_pred_scaled)
        sns.heatmap(cm, 
                     annot=True,
                     fmt="d",
                     cmap="Reds")
        plt.xlabel(f"Predicted {target}")
        plt.ylabel(f"Actual {target}")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        st.markdown(" - The **Confusion Matrix** shows the ratios of true positives, false postives, true negatives, and false negatives.")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#The following code is for the K nearest neighbor machine learning model 
elif model == "K-Nearest Neighbor": 
    st.subheader("If you uploaded your dataset:")
    st.markdown(" - Make sure your dataframe is formatted according to [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf). This is the best way to make sure that the model will work with your dataframe.")
    st.markdown(" - Convert any categorical variables into numeric variables (eg. 1 for true, 0 for false). K-Nearest Neighbor models only work with numeric type variables. This will help be more accurate, because it will have access to more features when making the predictions.")
    st.markdown(" - Account for any missing data. K-Nearest Neighbors models cannot handle missing data. Thus, due to the many possible datasets this model might encounter, the app atoumatically drops observations with missing data.")
    st.markdown("---") 
#----------------------------------------------------------------------------------------------------------------------------------------------------
#Here, we define the target and feature variables using the chosen dataset: 
    #This is code is the exact same as the code from the logistic regression model (just splitting the data into training and testing sets) 
    st.subheader("Defining target and feature variables")
    df_num = df.select_dtypes(include='number') #only numeric variables because of the requirement for lineaer regression. 
    drop_cols = st.multiselect(label="Drop any columns you do not be a part of the model as either features or targets (eg. if your dataset has a unique id label for each column it can be dropped here).", #the multi-selection box lets user drop any columns they do not want to be feature variables in the model. 
                             options= df_num.columns,
                             help="If your dataframe has any irrelevant columns, like a unique observation label, you can drop them by selecting it here.")
    df_rel = df_num.drop(df_num[drop_cols], axis=1)#dropping NaN so model can run 
    df_clean = df_rel.dropna()
    target = st.selectbox(label="Choose a Variable to Predict. Since logistic regression is a classification model, your target variable (what you put here) must be a categorical variable, but encoded as numeric.",  #the selection box lets users choose which variable they want to predict
                          options = df_clean.columns, 
                          help="Choose one of the numeric columns to be your target variable. This is what the model will be predicting.")
    y = pd.Series(df_clean[target])#defining the target variable that will be used in the model. 
    feature_vars = pd.DataFrame(df_clean)
    X = feature_vars.drop(target, axis=1) #defining the feature variables that will be used in the model. 
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #displaying both target and feature variables for the user.
    # This code is also the same as the other two models.
    features_column, target_column = st.columns([1,1], gap="small", vertical_alignment= "top")
    with features_column:
        st.write("Feature Variables") 
        st.write(X.head())
    with target_column:
        st.write("Target Variable")
        st.write(y.head())
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #Here the user can set a few hyperpermaters 
    #The training size and scale selection code is the same as the other two models here too. 
    st.subheader("Making the model")
    train_size = st.slider(label = "Choose the size of the training set", #allows the user choose the size of training set.  
                                  min_value=0.01,
                                  max_value=0.99, 
                                  value = 0.8)
    scale_selection = st.selectbox(label="Choose from scaled or unscaled data", #allows the user to choose between scaled and unscaled data.
                                   options= ["Unscaled", "Scaled"])
    st.markdown(" - For K-nearest neighbors models, scaling the data does affect the results (accuracy, for example), unlike linear and logistic regression.")
    neighbors_num = st.slider(label = "Choose the number of neighbors you want the model use.", 
                              min_value=1,
                              max_value=21,
                              value=5,
                              step=2)
    st.markdown(" - The accuracy of the model depends on how many neighbors the model is considering when classifying a point. Essentially, the model looks at 'k' number of nearest observations to the data point it is trying to predict. Whichever is the most common class out of the 'k' number of nearest neighbors will be the model's prediction. This number must be odd so there are no 'ties'.")
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #The following code initiates and trains the model and then makes predictions. (This is for unscaled, raw data).
    #It follows the same formula as linear and logistic regression.
    if scale_selection == "Unscaled":
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, #splitting the dataframe into training and testing sets 
                                                    train_size=train_size,
                                                    random_state=42)
        knn_raw = KNeighborsClassifier(n_neighbors=neighbors_num)#initiating the model based on how many neigbors the model has chosen
        knn_raw.fit(X_train_raw, y_train_raw)#training the model on the training data
        y_pred_raw = knn_raw.predict(X_test_raw)#making predictions based off of the trained model.
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #These are the metrics the app will use to evaluate the K-nearest neighbors regresssion model:  
        accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)  #creating the accuracy score
        classification_report_raw = classification_report(y_test_raw, 
                                                             y_pred_raw, 
                                                             output_dict=True) #creating the classification report
        cm = confusion_matrix(y_test_raw, y_pred_raw) #creating the confusion matrix    
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #displaying the evaluation metrics. 
        st.header("How did the model perform?")
        st.subheader("Accuracy")
        st.text(f"{accuracy_raw * 100:.2f}% of predictions were correct.")#displaying accuracy score
        st.subheader("Classification Report")
        st.dataframe(classification_report_raw)# Display classification report
        st.markdown(" - **Precision** is the number of true positives divided the total number of predicted positives.")
        st.markdown(" - **Recall** is the number of true positives divided by the sum of true positives and false negatives.")
        st.markdown(" - The **f1-score** is the harmonic mean of the precision and recall statistics.")
        fig, ax = plt.subplots()#initializing the heatmap that will display the confusion matrix results.
        sns.heatmap(cm, 
                annot=True,
                fmt="d",
                cmap="Reds")
        plt.xlabel(f"Predicted {target}")
        plt.ylabel(f"Actual {target}")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        st.markdown(" - The **Confusion Matrix** shows the ratios of true positives, false postives, true negatives, and false negatives.")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #The following code initiates and trains the model and then makes predictions. (This is for unscaled, raw data).
    #It follows the same formula as linear and logistic regression.
    #It is almost the exact same as the code for unscaled features. 
    elif scale_selection == "Scaled":
        scaler = StandardScaler() #initializing a standard scaler to scale features 
        features_scaled = scaler.fit_transform(X)#scaling the features
        X_scaled = pd.DataFrame(features_scaled, columns=X.columns) #putting the scaled data in a df
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, #splitting the dataframe into training and testing sets 
                                                                    train_size= train_size, 
                                                                    random_state=42)
        knn_scaled = KNeighborsClassifier(n_neighbors=neighbors_num)#initializing the model
        knn_scaled.fit(X_train_scaled, y_train_scaled)#training the model
        y_pred_scaled = knn_scaled.predict(X_test_scaled)#making predictions on the testing data using the model
#----------------------------------------------------------------------------------------------------------------------------------------------------
        #These are the metrics the app will use to evaluate the K-nearest neighbors regresssion model:  
        accuracy_scaled = accuracy_score(y_test_scaled, y_pred_scaled)  #creating the accuracy score
        classification_report_scaled = classification_report(y_test_scaled, 
                                                             y_pred_scaled, 
                                                             output_dict=True) #creating the classification report
        cm = confusion_matrix(y_test_scaled, y_pred_scaled) #creating the confusion matrix    
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #displaying the evaluation metrics. 
        st.header("How did the model perform?")
        st.subheader("Accuracy")
        st.text(f"{accuracy_scaled * 100:.2f}% of predictions were correct.")#displaying accuracy score
        st.subheader("Classification Report")
        st.dataframe(classification_report_scaled)# Display classification report
        st.markdown(" - **Precision** is the number of true positives divided the total number of predicted positives.")
        st.markdown(" - **Recall** is the number of true positives divided by the sum of true positives and false negatives.")
        st.markdown(" - The **f1-score** is the harmonic mean of the precision and recall statistics.")
        fig, ax = plt.subplots()#initializing the heatmap that will display the confusion matrix results.
        sns.heatmap(cm, 
                annot=True,
                fmt="d",
                cmap="Blues")
        plt.xlabel(f"Predicted {target}")
        plt.ylabel(f"Actual {target}")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        st.markdown(" - The **Confusion Matrix** shows the ratios of true positives, false postives, true negatives, and false negatives.")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#The following code is for the Decision Tree Machine Learning Model!
#a lot of the code is similar to the previous three machine learning models.
elif model == "Decision Tree":
    st.subheader("If you uploaded your dataset:")
    st.markdown(" - Make sure your dataframe is formatted according to [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf). This is the best way to make sure that the model will work with your dataframe.")
    st.markdown(" - Convert any categorical variables into numeric variables (eg. 1 for true, 0 for false). K-Nearest Neighbor models only work with numeric type variables. This will help be more accurate, because it will have access to more features when making the predictions.")
    st.markdown(" - Account for any missing data. K-Nearest Neighbors models cannot handle missing data. Thus, due to the many possible datasets this model might encounter, the app atoumatically drops observations with missing data.")
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#Here, we define the target and feature variables using the chosen dataset: 
    #This is code is the exact same as the code from the other three regression model (just splitting the data into training and testing sets) 
    st.subheader("Defining target and feature variables")
    df_num = df.select_dtypes(include='number') #only numeric variables because of the requirement for lineaer regression. 
    drop_cols = st.multiselect(label="Drop any columns you do not be a part of the model as either features or targets (eg. if your dataset has a unique id label for each column it can be dropped here).", #the multi-selection box lets user drop any columns they do not want to be feature variables in the model. 
                             options= df_num.columns,
                             help="If your dataframe has any irrelevant columns, like a unique observation label, you can drop them by selecting it here.")
    df_rel = df_num.drop(df_num[drop_cols], axis=1)#dropping NaN so model can run 
    df_clean = df_rel.dropna()
    target = st.selectbox(label="Choose a Variable to Predict. Since DecisionTreeClassifier is a classification model, your target variable (what you put here) must be a categorical variable, but encoded as numeric.",  #the selection box lets users choose which variable they want to predict
                          options = df_clean.columns, 
                          help="Choose one of the numeric columns to be your target variable. This is what the model will be predicting.")
    y = pd.Series(df_clean[target])#defining the target variable that will be used in the model. 
    feature_vars = pd.DataFrame(df_clean)
    X = feature_vars.drop(target, axis=1) #defining the feature variables that will be used in the model.
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #displaying both target and feature variables for the user.
    # This code is also the same as the other three models.
    features_column, target_column = st.columns([1,1], gap="small", vertical_alignment= "top")
    with features_column:
        st.write("Feature Variables") 
        st.write(X.head())
    with target_column:
        st.write("Target Variable")
        st.write(y.head())
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #Here the user can set a few hyperpermaters 
    #The training size and scale selection code is the same as the other two models here too. 
    st.subheader("Making the model")
    train_size = st.slider(label = "Choose the size of the training set", #allows the user choose the size of training set.  
                                  min_value=0.01,
                                  max_value=0.99, 
                                  value = 0.8)
    max_depth = st.slider(label = "Choose the maximum depth (the longest path from the first question to a final split) for the decision tree.",
                          min_value=2,
                          max_value=6,
                          value=4)
    min_sample_split = st.slider(label="Choose the minimum number of samples required to split a node",
                                 min_value=2, 
                                 max_value=8,
                                 value=5,
                                 help="If the number of samples in a node is smaller than the selected number, that node will be a leaf node.")
    min_samples_leaf  = st.slider(label="Choose the minimum number of samples required to be at a leaf node.", 
                                  min_value=1,
                                  max_value=4,
                                  value=2,
                                  help="A split point at any depth will only be considered if it leaves at least the selected number training samples in each branch.")
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #The following code is for the initializing, training and making predictions with the decision tree model. 
    #Here we employ the hyperperameters chosen by the user. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, #splitting the dataframe into training and testing sets 
                                                        train_size= train_size, 
                                                        random_state=42)
    dtree = DecisionTreeClassifier(max_depth=max_depth,
                                   min_samples_split=min_sample_split,
                                   min_samples_leaf=min_samples_leaf)#initializing the model
    dtree.fit(X_train, y_train)#training the model
    y_pred = dtree.predict(X_test)#making predictions on the testing data using the model
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #the following code displays the actual decision tree.
    st.subheader("Your Decision Tree!")
    dot_data = tree.export_graphviz(dtree, feature_names=X_train.columns,
                                    filled=True)
    graph = graphviz.Source(dot_data)#makes the decision tree graph
    graph #displays the decision tree graph
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#These are the metrics the app will use to evaluate the K-nearest neighbors regresssion model:  
    accuracy = accuracy_score(y_test, y_pred)  #initiating the accuracy score
    classification_report = classification_report(y_test, 
                                                  y_pred, 
                                                  output_dict=True) #initiating the classification report
    cm = confusion_matrix(y_test, y_pred) #initiating the confusion matrix    
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #displaying the evaluation metrics. 
    st.header("How did the model perform?")
    st.subheader("Accuracy")
    st.text(f"{accuracy * 100:.2f}% of predictions were correct.")#displaying accuracy score
    st.subheader("Classification Report")
    st.dataframe(classification_report)# Display classification report
    st.markdown(" - **Precision** is the number of true positives divided the total number of predicted positives.")
    st.markdown(" - **Recall** is the number of true positives divided by the sum of true positives and false negatives.")
    st.markdown(" - The **f1-score** is the harmonic mean of the precision and recall statistics.")
    fig, ax = plt.subplots()#initializing the heatmap that will display the confusion matrix results.
    sns.heatmap(cm, 
                annot=True,
                fmt="d",
                cmap="Reds")
    plt.xlabel(f"Predicted {target}")
    plt.ylabel(f"Actual {target}")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
    st.markdown("---")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    #The following code is for specific decision tree metrics (ROC and AUC)
    st.subheader("Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC)")
    y_probs = dtree.predict_proba(X_test)[:, 1]# Get the predicted probabilities for the positive class (survival)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs) # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    roc_auc = roc_auc_score(y_test, y_probs)# Compute the Area Under the Curve (AUC) score
    st.text(f"ROC AUC Score: {roc_auc:.2f}")
#----------------------------------------------------------------------------------------------------------------------------------------------------
    # Plot the ROC curve
    fig2, ax2 = plt.subplots()#initializing the ROC AUC graph
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') # Plotting 50% line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    sns.set_style(style="darkgrid")
    st.pyplot(fig2)
    st.markdown("The **ROC AUC Curve** plots the True Positive Rate (TPR) versus the False Positive Rate (FPR). Lowering the threshold decreases FPR but might lower TPR. Essentially the **ROC AUC Curve** visualizes the trade-off between TPR and FPR. It is also worth noting, any AUC score worse than 0.5 is worse than a 50-50 guess. ")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#Here ends the code 
