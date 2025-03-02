import pandas as pd            # Library for data manipulation
import seaborn as sns          # Library for statistical plotting
import matplotlib.pyplot as plt  # For creating custom plots
import streamlit as st         # Framework for building interactive web apps

# ================================================================================
#Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv("titanic.csv")

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.write(df.shape)
st.dataframe(df.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
st.write("**Number of Missing Values by Column**")
st.dataframe(df.isnull().sum()) 
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.


# ------------------------------------------------------------------------------
# Visualize Missing Data
# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.
st.subheader('Heatmap of Missing Values')
fig, ax = plt.subplots()
sns.heatmap(df.isnull(),cmap = 'inferno', cbar = False)
st.pyplot(fig)


# ================================================================================
# Interactive Missing Data Handling
#
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================
st.subheader('Handling Missing Data')

# Work on a copy of the DataFrame so the original data remains unchanged.
column = st.selectbox("Choose a column to fill", df.select_dtypes(include=["number"]).columns)
# Apply the selected method to handle missing data.
#st.dataframe(df[column_choice])

method = st.radio("Choose a method", options =
         ["Original DF", "Drop Rows", "Drop Columns with over 50% missing data ", "Impute Mean", "Impute Median", "Impute Zero"])
# Copy our original data frame; df_clean will be the modified df; df is not be touched 
df_clean = df.copy()

if method == "Original DF": 
    pass 
elif method == "Drop Rows": 
    df_clean = df_clean.dropna(subset= [column])
elif method == "Drop Columns with over 50% missing data ": 
    df_clean = df_clean.drop(columns = df_clean.columns[df_clean.isnull().mean() > .5])
elif method == "Impute Mean": 
    df_clean[column] = df_clean[column].fillna(df[column].mean())
elif method == "Impute Median": 
    df_clean[column] = df_clean[column].fillna(df[column].median())
elif method == "Impute Zero": 
    df_clean[column] = df_clean[column].fillna(0)
# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned
#
#st.dataframe(df_clean)
st.subheader("Cleaned Data Distribution")
fig, ax = plt.subplots()
sns.histplot(df_clean[column], kde=True)
st.pyplot(fig)
st.write(df_clean.describe())
#imputation methods should generally not be used for instances of missing not at random
# Display side-by-side histograms and statistical summaries for the selected column.
# ------------------------------------------------------------------------------

