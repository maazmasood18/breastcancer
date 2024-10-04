import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Load data function with updated path
def load_data():
    data = pd.read_csv('gbsg.csv')  # Update the path to your CSV file
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data.drop(['pid', 'rfstime'], axis=1, inplace=True)
    return data

# Train the Random Forest model
def train_model(data):
    features = ['age', 'meno', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon']
    X = data[features]
    y = data['status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train RandomForest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=52)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, confusion, report

# Streamlit app layout
st.title('Breast Cancer Detection using Random Forest Classifier')
st.write('This app trains a Random Forest Classifier to predict breast cancer status.')

# Load and display data
data = load_data()
st.write('Dataset:')
st.dataframe(data)

# Define X and y for visualization
features = ['age', 'meno', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon']
X = data[features]
y = data['status']

# Plot pairplot
st.subheader('Data Visualization')
sns.pairplot(pd.concat([X, y], axis=1), hue='status', kind='hist')
st.pyplot()

# Heatmap of correlations
st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis
sns.heatmap(data.corr(), annot=True, ax=ax)  # Pass the axis to Seaborn
st.pyplot(fig)  # Pass the figure to Streamlit

# Train and display results
if st.button('Train Model'):
    st.write('Training the Random Forest model...')
    accuracy, confusion, report = train_model(data)
    
    st.write(f'**Accuracy**: {accuracy}')
    st.write('**Confusion Matrix**:')
    st.write(confusion)
    st.write('**Classification Report**:')
    st.text(report)
