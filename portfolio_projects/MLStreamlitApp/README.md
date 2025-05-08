## The Machine Learning Experience!
##### In this project, I was asked to make a streamlit app where users could upload their own dataset and run a supervised machine learning model on it. Users can select between four type of supervised machine learning models, and change hyperpermaters. I was then asked to publish this app to the streamlit cloud. 


##### In my app, "The Machine Learning Experience", the user can upload their own dataset, or choose from three preselected ones (Iris, Titanic, and Penguins). Then they can choose between: 
- ##### `LinearRegression`: machine learning model the predicts the value of a variables based on the other varialbes. It does this by drawing a line of best fit through the data that minimizes the residuals using the training data. 
- ##### `LogisticRegression`: machine learning model that typically is used to classify binary data. It makes predictions by using their feature variables.
- ##### `KNeighborsClassifer`: machine learning model that is typically used for variable classification. The model classifies the observation by taking a 'majority vote' the k-nearest observations' classifications. 
- ##### `DecisionTreeClassifier`: a machine learning model that uses a set of sequential questions to split the data into similar groups which it then uses to make predictions. 

#

### The app returns metrics and graphs specific to the chosen model. 

<img align="left" width="200" height="150" 
   alt="Screenshot 2025-05-08 at 12 52 01 AM" src="https://github.com/user-attachments/assets/51c5aa76-7894-4c81-840c-fe37a9465285" />
<br>
##### `LinearRegression` returns the Mean Squared Error, Root Mean Squared Error, R^2 Score, Coefficients and Intercept. 
<br>
<br>
<br>
<img align="left" width="200" height="150" 
   alt="Screenshot 2025-05-08 at 12 54 45 AM" src="https://github.com/user-attachments/assets/1866f71e-e803-4e3d-a1ae-b29eaf3af743" />
<br>

##### `LogisticRegression` returns Accuracy, Precision, Recall, f1-score, Coefficients and Intercept 
<br>
<br>
<br>

<img align="left" width="200" height="150" 
   alt="Screenshot 2025-05-08 at 12 58 30 AM" src="https://github.com/user-attachments/assets/84e25ea1-30c0-4e02-84cd-96e2c62bb15a" />
<br>

##### `KNeighborsClassifer` returns Accuracy, Precision, Recall, and the f1-score. 
<br>
<br>
<br>

<img align="left" width="200" height="100" 
   alt="Screenshot 2025-05-08 at 1 00 57 AM" src="https://github.com/user-attachments/assets/96c5607f-5732-4f1c-95b3-6a0cb1bf1045" />

##### `DecisionTreeClassifier` returns Accuracy, Precision, Recall, f1-score, a Confusion Matrix, and Receiver Operating Charateristic (ROC) curve. 
<br>


#

##### The nuts and bolts:
- ##### Required Libraries: `graphviz 0.20.1`, `matplotlib 3.10.1`, `numpy 2.2.4`, `pandas 2.2.3`, `scikit_learn 1.6.1`, `seaborn 0.13.2`, `streamlit 1.37.1` and `streamlit_option_menu 0.4.0`.
   - ##### There is also a `requirements.txt` file for further clarity.
- ##### If you want to run the file locally, you can download the .py file and run the app using `streamlit run MLStreamlitApp.py` in the terminal. You can also run the app on the [streamlit cloud](https://jtmachinelearning.streamlit.app)!   
