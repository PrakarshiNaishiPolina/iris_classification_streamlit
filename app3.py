import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Classification", page_icon="💮")

st.markdown(
    """
    <style>
    /* General body styling */
    .stApp {
        background-color: #f7f7f7;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Title and header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #3E2C41;
    }
    
    /* Custom slider styling (Streamlit’s internal  */
    .stSlider > div {
        background-color: #eaeaea;
        border-radius: 5px;
    }
    .stSlider .css-1cpxqw2, .stSlider .css-1thv13d {
        background: #3E2C41;
    }
    
    /* Success messages */
    .st-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Customize Matplotlib figures (if needed) */
    .stPlotlyChart, .css-1x8cf1d {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
    
    
    .big-font {
        font-size: 30px !important;
        color:rgb(114, 46, 223);
        text-align: center;
        margin-top: 20px;
    }
   
    </style>
    """,
    unsafe_allow_html=True
)




iris = load_iris()
X, y = iris.data, iris.target
species = iris.target_names  # e.g., ["setosa", "versicolor", "virginica"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


st.title("🌸 Iris Flower Classification")
st.write("Enter the petal and sepal measurements to predict the species.")

# User input sliders with min, max, and default values
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Convert input to a NumPy array
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Make prediction

prediction = model.predict(input_features)[0]  # Extract the first (and only) prediction
prediction_proba = model.predict_proba(input_features)  # Returns a nested list

# Display the prediction and confidence
st.success(f"Predicted Species: **{species[prediction]}**")
st.write(f"Confidence: {prediction_proba[0][prediction] * 100:.2f}%")


st.subheader("Iris Dataset Visualization")
fig, ax = plt.subplots()
sns.scatterplot(
    x=iris.data[:, 0],
    y=iris.data[:, 1],
    hue=[species[i] for i in iris.target],
    palette="viridis",
    ax=ax
)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
st.pyplot(fig)


st.markdown('<p class="big-font">Enjoy exploring the Iris dataset!</p>', unsafe_allow_html=True)




