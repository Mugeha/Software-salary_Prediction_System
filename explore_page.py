import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Helper function to shorten categories
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

# Helper function to clean experience data
def clean_experience(x):
    if isinstance(x, str):
        if x == 'More than 50 years':
            return 50.0
        if x == 'Less than 1 year':
            return 0.5
        return float(x.replace(' years', '').replace(' year', ''))
    return x

# Helper function to clean education data
def clean_education(x):
    if "Bachelor's degree" in x:
        return "Bachelor's degree"
    if "Master's degree" in x:
        return "Master's degree"
    if "Professional degree" in x or "Other doctoral" in x:
        return "Post grad"
    return "Less than a Bachelors"

# Function to load and clean the data
@st.cache_data
def load_data():
    df = pd.read_csv('unzippedstack/survey_results_public.csv')
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]

    df = df[df["ConvertedComp"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedComp"] <= 250000]
    df = df[df["ConvertedComp"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedComp": "Salary"}, axis=1)
    return df

# Load the data using the defined function
df = load_data()

# Train the LabelEncoder on the entire dataset
le_country = LabelEncoder()
le_education = LabelEncoder()

# Fit the label encoders
df['Country'] = le_country.fit_transform(df['Country'])
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])

# Splitting data into features and target variable
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Save the trained models and encoders
pickle.dump(regressor, open('model.pkl', 'wb'))
pickle.dump(le_country, open('le_country.pkl', 'wb'))
pickle.dump(le_education, open('le_education.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Function to display the explore page
def show_explore_page():
    st.title("Explore Software Engineer Salaries")

    st.write("""
    ### Stack Overflow Developer Survey 2020
    """)

    # Pie chart of data from different countries
    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=le_country.inverse_transform(data.index), autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")

    st.write("#### Number of Data from Different Countries")
    st.pyplot(fig1)

    # Bar chart of mean salary based on country
    st.write("#### Mean Salary Based On Country")
    country_salary_data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(country_salary_data)

    # Line chart of mean salary based on years of professional coding experience
    st.write("#### Mean Salary Based On Years of Professional Coding Experience")
    experience_salary_data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(experience_salary_data)

# Display the explore page
show_explore_page()
