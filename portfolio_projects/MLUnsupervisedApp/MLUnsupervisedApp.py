# Jack Thomas 
# Portfolio Update 3 for MDSC 20009 
# 4/14/25

#importing relevant libraries.
import streamlit as st 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing relevant machine learning models from sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#other important things that need to work
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Unsupervisde Machine Learning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#The following code is for the sidebar 
st.sidebar.header("The Sidebar!", divider = "red")
with st.sidebar: 
    page = option_menu(menu_title="", 
                       options = ["Dataset Preprocessing", "Models"],
                       icons = ["nut-fill", "bar-chart-fill"], #icons come from https://icons.getbootstrap.com/?q=tree
                       default_index=0,
                       orientation="vertical")
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#User can select from four pre-uploaded dataframe, using a selection box. 
st.sidebar.markdown("---")
st.sidebar.subheader("Loading Dataset and Model", 
                     divider = "red")
selection = st.sidebar.selectbox(label= "Select Your Dataset", 
                                 options= ["NBA Players", "NBA Teams", "Other"]) #selection box with each dataframe.
#-------------------------------------------------------------------------------------------------------------------------------------------------------
if selection == "NBA Players": 
    df = pd.read_csv(r"Data\NBA Stats 202425 All Metrics  NBA Player Props Tool.csv")
    df.drop("RANK", axis = 1, inplace=True)
elif selection == "NBA Teams": 
    df = pd.read_csv(r"Data\NBA Team Stats.csv")
    df.drop(['Rk', "Unnamed: 17", "Unnamed: 22", "Unnamed: 27", "Arena"], axis = 1, inplace=True)
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
#-----------------------------------------------------------------------------------------------------------------------------------------------------
model = st.sidebar.selectbox(label="Select Model",
                             options= ["Principle Component Analysis", #options defines each page (what machine learning models the app will offer).
                                  "K-means clustering"])
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#an if statement that iterates over "selection" and reads the right dataframe. 
#New object called file_select stores the users choice.  
if page == "Models":
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#The following code is for the PCA Hyperameters 
    if model == "Principle Component Analysis":
        st.sidebar.subheader("Tune your hyperparameters", divider="red")
        no_numeric_data = st.sidebar.selectbox(label="Include Non-Numeric Data",
                                                options=["No", "Yes"],
                                                help="It is recommended that you encode your categorical data into numeric variables for optimal model performance.")
        drop_cols = st.sidebar.multiselect(label="Drop any columns not part of analysis.", #the multi-selection box lets user drop any columns they do not want to be feature variables in the model. 
                                options= df.columns,
                                help="If your dataframe has any irrelevant columns (e.g. a unique observation label) or columns you do not want the model to include, you can drop them by selecting them here.")
        components = st.sidebar.slider(label="Number of Components", 
                                    min_value=1, 
                                    max_value=15, 
                                    value=2)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#The following code runs the PCA Model: 
#Preprocessing the Dataset 
        df_num = df.select_dtypes(include='number') #This df includes only numeric variables
        if no_numeric_data == "No":
            df_1 = df_num
        elif no_numeric_data == "Yes": 
            df_1 = df
        df_clean = df_1.drop(df_1[drop_cols], axis=1)#allows the user to drop any irrelevant colomns
        df_clean.dropna(inplace=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Defining target, feature variables 
        target = st.sidebar.selectbox(label="Choose a Variable to Observe:",  #the selection box lets users choose which variable they want to predict
                            options = df_clean.columns, 
                            help="Choose one of the numeric columns to be your target variable. This is what the model will be predicting. Be sure to input it exactly as it appears in the uploaded file.")
        y = pd.Series(df_clean[target])#defining the target variable that will be used in the model. 
        feature_vars = pd.DataFrame(df_clean)
        X = feature_vars.drop(target, axis=1)#defining the feature variables that will be used in the model. 
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Putting the feature variables on the same scale  
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Fitting the model
        pca = PCA(n_components=components)
        X_pca = pca.fit_transform(X_std)    
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Displaying metrics and graphs for the model
#will be displayed in these columns
        col1, col2 = st.columns([2,3], gap = "medium", vertical_alignment= "top")
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Displaying the variance metrics    
        with col1: 
            # Display the proportion of variance explained by each component
            st.subheader("Variance", divider="red")
            explained_variance = pca.explained_variance_ratio_
            st.write("Explained Variance Ratio:")
            with st.container(border=True):
                st.text(explained_variance)
            st.write("Cumulative Explained Variance:")
            with st.container(border=True):
                st.text(np.cumsum(explained_variance))
            st.write("The Explained Variance Ration shows the amount of variance each principal component explains, starting from the first principal component. The cumulative explained variance is the cumulative sum of variance explained by the component starting from the first. (It is the sum of the individual explained variance ratios).")  
            st.write("---")
            pca_full = PCA(n_components = 15).fit(X_std)
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            fig2, ax2 = plt.subplots()
            plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Variance Explained')
            plt.xticks(range(1, len(cumulative_variance)+1))
            plt.grid(True)
            st.pyplot(fig2)
            st.write("The above plot visualizes the cumulative explained variance of the model. As you increase the principal components on the x-axis, the total variance explained increases by the amount displayed on the graph.")     
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Creating the 2-Dimension PCA for the graph
        pca_graph = PCA(n_components=2)
        X_pca_graph = pca_graph.fit_transform(X_std)
        df_pca = pd.DataFrame(data = X_pca_graph,
                            columns=["PCA1", "PCA2"])
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#displaying a graph of the PCA (Has to be 2d, so only two components)
        with col2:
            st.subheader("Visualization", divider = "red")
            fig, ax = plt.subplots()
            loadings = pca_graph.components_.T
            scaling_factor = 50.0
            sns.scatterplot(x = "PCA1",
                        y = "PCA2", 
                        data = df_pca, 
                        hue = df[target])
            sns.set_style(style="darkgrid")
            plt.title(f"PCA: 2D Project of {target}")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(loc='best')
            st.pyplot(fig)
            st.write("This scatterplot plots the data on a two-diminesional plane (only two principal components). You can see the distribution of the data and if any groups forms.")
#-------------------------------------------------------------------------------------------------------------------------------------------------------
 #The following code is for the K-means clustering model
 #Here the user can choose their hyperperameters
    elif model == "K-means clustering":
        st.sidebar.subheader("Tune your hyperparameters", divider="red")
        no_numeric_data = st.sidebar.selectbox(label="Include Non-Numeric Data",
                                                options=["No", "Yes"],
                                                help="It is recommended that you encode your categorical data into numeric variables for optimal model performance.")
        drop_cols = st.sidebar.multiselect(label="Drop any columns not part of analysis", #the multi-selection box lets user drop any columns they do not want to be feature variables in the model. 
                                options= df.columns,
                                help="If your dataframe has any irrelevant columns (e.g. a unique observation label) or columns you do not want the model to include, you can drop them by selecting them here.")
        k = st.sidebar.slider(label="Choose the number clusters", 
                                     min_value = 2,
                                     max_value= 10,
                                     value=5)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#The following code runs the PCA Model: 
#Preprocessing the Dataset 
        df_num = df.select_dtypes(include='number') #This df includes only numeric variables
        if no_numeric_data == "No":
            df_1 = df_num
        elif no_numeric_data == "Yes": 
            df_1 = df
        df_clean = df_1.drop(df_1[drop_cols], axis=1)#allows the user to drop any irrelevant colomns
        df_clean.dropna(inplace=True)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Defining target, feature variables 
        target = st.sidebar.selectbox(label="Choose a Variable to Observe:",  #the selection box lets users choose which variable they want to predict
                            options = df_clean.columns, 
                            help="Choose one of the numeric columns to be your target variable. This is what the model will be predicting. Be sure to input it exactly as it appears in the uploaded file.")
        y = pd.Series(df_clean[target])#defining the target variable that will be used in the model. 
        feature_vars = pd.DataFrame(df_clean)
        X = feature_vars.drop(target, axis=1)#defining the feature variables that will be used in the model. 
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Putting the feature variables on the same scale  
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Fitting the model
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(X_std)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Displaying metrics and graphs for the model
        st.header("K-means clustering!", divider="red")
        st.write("The first visualization plots the data across two principal components. The different colors represent the predicted clusers from the K-means clustering model. The second visualization plots the same data across two principal components. In this chart the different colors represent the true classifications. Compare the K-means clusters to the real labels and evaluate how the model performed!")
#will be displayed in these columns
        col1, col2,  = st.columns([2,2], gap = "large", vertical_alignment="top")
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Creating the 2-Dimension PCA for the graph
        pca_graph = PCA(n_components=2)
        X_pca_graph = pca_graph.fit_transform(X_std)
        df_pca = pd.DataFrame(data = X_pca_graph,
                            columns=["PCA1", "PCA2"])
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Displaying the K-means cluster graphs  
        with col1:
            fig, ax = plt.subplots()
            loadings = pca_graph.components_.T
            scaling_factor = 50.0 
            sns.scatterplot(x = "PCA1",
                        y = "PCA2", 
                        data = df_pca, 
                        hue = clusters)
            sns.set_style(style="darkgrid")
            plt.title(f"K-means clustering predictions (2D PCA)")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(loc='best')
            st.pyplot(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Graphing the true labels (same code as above but with different hue)
        with col2:
            fig, ax = plt.subplots()
            loadings = pca_graph.components_.T
            scaling_factor = 50.0 
            sns.scatterplot(x = "PCA1",
                        y = "PCA2", 
                        data = df_pca, 
                        hue = df[target])
            sns.color_palette('husl', 8)
            sns.set_style(style="darkgrid")
            plt.title(f"True {target}: (2D PCA)")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(loc='best')
            st.pyplot(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Displaying the centroid information
        st.subheader("Centroids", divider="red")
        st.write("Centroid Locations")
        with st.container(height= 150,border=True):
            st.text(kmeans.cluster_centers_)
        st.write("First 10 Centroid Assingments")
        with st.container(border=True):
            st.text(clusters[:10])
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculating and then displaying the optimal centroid number and then graphing the results       
        st.subheader("Best Number of Clusters", divider="red")
        ks = range(2, 11) #can try any number of clusters from two to ten
        wcss = [] # Within-Cluster Sum of Squares for each k
        silhouette_scores = [] # Silhouette scores for each k
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_std)
            wcss.append(km.inertia_)  # inertia: sum of squared distances within clusters
            labels = km.labels_
            silhouette_scores.append(silhouette_score(X_std, labels))
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Graphing the WCSS per number of clusters 
        col3, col4 = st.columns([2,2], gap = "large", vertical_alignment="top")
        with col3:
                fig, ax = plt.subplots()
                plt.plot(ks, wcss, marker = "o", color="lightcoral")
                plt.xlabel('Number of clusters (k)')
                plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
                plt.title('Number of K vs. WCSS')
                sns.set_style(style="darkgrid")
                st.pyplot(fig)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Graphing the Silhouette Score per number of clusters (same as above)
        with col4:
            fig2, ax2 = plt.subplots()
            plt.plot(ks, silhouette_scores, marker = "o", color="orange")
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Silhouette Scores')
            plt.title('Number of K vs. Silhouette Scores')
            sns.set_style(style="darkgrid")
            st.pyplot(fig2)          

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Displaying the acutal WCSS and Silhouette scores 
        st.write("Within-Cluster Sum of Squares")
        with st.container(border=True): 
            st.text(wcss)
        st.write("Silhouette Scores")
        with st.container(border=True):
            st.text(silhouette_scores)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# The following code is the for the nuts and bolts section  
elif page == "Dataset Preprocessing": 
    st.header("Unsupervised Machine Learning!", divider="red")
    st.write("This streamlit app allows you to train unsupervised machine learning models onto a dataset of your choise. In the sidebar, you can either upload your own dataset or choose from one of two pre-uploaded datasets that have stats from the 2024-2025 NBA season. You also have the options to choose your desired unsupervised machine learning model (Principal Component Analysis or K-Means Clustering), and set your own hyperperameters.")
    st.subheader("Nuts and Bolts of Uploading your own dataset", divider = "grey")
    st.markdown("#### If you uploaded your dataset:")
    st.markdown(" - Make sure your dataset is formatted according to [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf). This is the best way to make sure that the models will work with your dataset.")
    st.markdown(" - Encode any categorical variables into numeric variable (e.g. set a binary categorical variable to 1s and 0s). The machine learning models on this app only accept numeric variables. While there is an option for you to drop all categorical variables, they often have valuable information, and thus, I recommend encoding variables beforehand. If there are categorical variables in your dataset, you can drop them using the option in the sidebar")
    st.markdown(" - Account for any missing data. Due to the many possible datasets this model might encounter, the app atoumatically drops observations with missing data.")
    with st.container(height = 400, border=True): 
        st.markdown("#### Your dataset!")
        st.write(df)
 
    
    
