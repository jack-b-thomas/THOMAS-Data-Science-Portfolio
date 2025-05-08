## Unsupervised Machine Learning: NBA Edition! 


#### For my final portfolio project, I created an unsupervised machine learning app on streamlit. Within the app, the user can upload a data set, or choose from two datasets with player and team stats from the 2024-2025 NBA season.

#### To get the app working, go to [the streamlit cloud](https://jtunsupervisedmachinelearning.streamlit.app) or download the file and locally host the app, using `streamlit run MLUnsupervisedApp.py` in the terminal. (Note, sometimes when I locally hosted the app, it needed a different file path for the NBA datsets, which has been commented right next to the cloud file path code).

#### The app requires the following libraries: `matplotlib 3.10.1`, `numpy 2.2.5`, `pandas 2.2.3`, `scikit_learn 1.6.1`, `seaborn 0.12.2`, `streamlit 1.37.1`, and `streamlit_option_menu 0.4.0` 
#

##### If YOU choose Principal Component Analysis (PCA) - where you reduce the number of variables into components.
- ##### You can choose the number of components
- ##### You can see a visualization that shows the data graphed with repsect to two principal components
- ##### You can see the variance statistics
- ##### You can the cumulative variance plotted with respect to the number of components

##### If YOU choose K-means Clustering (Kmeans) - where you group the number of observations into clusters
- ##### You can choose the number of clusters
- ##### You can see the clusters graphed with respect to two principal components
- ##### You can compare the clusters to the true labels (which is graphed in the same way as the clusters).
- ##### You can see centroid statistics
- ##### You can see the Within Cluster Sum of Squares and Silhouette Scores graphed with respect to the number of clusters

#

#### Principal Component Analysis Visualization Example:
<img  width="250" height="200" alt="Screenshot 2025-05-08 at 12 08 59 AM" src="https://github.com/user-attachments/assets/abb1df23-7b57-47c4-8cbe-b844d5ca563b" />

# 


#### K-Means Clustering Visualization Example:

<img width="612" alt="Screenshot 2025-05-08 at 12 08 36 AM" src="https://github.com/user-attachments/assets/9278632b-ef03-4451-8375-9dc46aa81a6d" />







<p style= 'text-alling: center; color: grey; font-size:8px'> NBA Player Data comes from <a href ='https://www.nbastuffer.com'> NBA Stuffer</a>. NBA Team Data comes from <a href ='https://www.basketball-reference.com/leagues/NBA_2025.html'> Basketball Reference</a>.

