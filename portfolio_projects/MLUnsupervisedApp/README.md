## Unsupervised Machine Learning: NBA Edition! 


#### For my final portfolio project, I created an unsupervised machine learning app on streamlit. Within the app, the user can upload a data set, or choose from two datasets with player and team stats from the 2024-2025 NBA season.

#### To get the app working, go to [the streamlit cloud](https://jtunsupervisedmachinelearning.streamlit.app) or download the file and locally host the app, using `streamlit run MLUnsupervisedApp.py` in the terminal. (Note, sometimes when I locally hosted the app, it needed a different file path for the NBA datsets, which has been commented right next to the cloud file path code).

#### The app requires the following libraries: `matplotlib 3.10.1`, `numpy 2.2.5`, `pandas 2.2.3`, `scikit_learn 1.6.1`, `seaborn 0.12.2`, `streamlit 1.37.1`, and `streamlit_option_menu 0.4.0` 
#

##### In this app, the user can choose from two different unsupervised machine learning models: 
- ##### `PCA` from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). This is the Principal Component Analysis unsupervised machine learning model where the model uses singular value decomposition to project the data with less dimensions (variables).
  - ##### You can choose the number of components
  - ##### You can see a visualization that shows the data graphed with repsect to two principal components
  - ##### You can see the variance statistics
  - ##### You can the cumulative variance plotted with respect to the number of components
- ##### `KMeans` from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). This is the K-means clustering unsupervised machine learning model where the model groups the data into 'clusters' based on shared characteristics. 
  - ##### You can choose the number of clusters
  - ##### You can see the clusters graphed with respect to two principal components
  - ##### You can compare the clusters to the true labels (which is graphed in the same way as the clusters).
  - ##### You can see centroid statistics
  - ##### You can see the Within Cluster Sum of Squares and Silhouette Scores graphed with respect to the number of clusters

#

#### The app returns metrics and graphs specific to the chosen model. 


<img align="left" width="300" height="240" alt="Screenshot 2025-05-08 at 12 08 59 AM" src="https://github.com/user-attachments/assets/abb1df23-7b57-47c4-8cbe-b844d5ca563b" />
<br>
<br>

##### `PCA` returns Explained Variance Ratio, Cumulative Explained Variance Ration, as well a 2-dimensional graph of the data. 
<br>
<br>
<br>
<br>
<br>


<img align="left" width="310" height="200" 
  alt="Screenshot 2025-05-08 at 1 52 04 AM" src="https://github.com/user-attachments/assets/810720de-0bfd-4431-b3cc-bb4edf20d363" />


##### `KMeans` returns Centroid Locations, First 10 Centroid Assignments, Within-Cluster Sum of Squares, Silhouette Scores, and comparisons between the models cluster predictions and the true lables, both graphed using 2-dimensional PCA.

<br>
<br>

#

#### NBA Player Data comes from [NBA Stuffer](https://www.nbastuffer.com). NBA Team Data comes from [Basketball Reference](https://www.basketball-reference.com/leagues/NBA_2025.html).

