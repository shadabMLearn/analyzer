
import streamlit as st
import pandas as pd
from recommendation_model import RecommendationModel
import pygwalker as pyg
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
import streamlit.components.v1 as components
from io import BytesIO
import numpy as np
import sketch
import pickle

# Set page configuration
st.set_page_config(
    page_title="Recommendation System App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the pre-trained model
model = RecommendationModel.load_model("bundle_recommendation_model.pkl")

# Streamlit app title with styling
st.title("🚀 Recommendation System App")
st.markdown("---")

# Initialize pygwalker communication
init_streamlit_comm()

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded!")

# Get bundle name from user input
bundle_name = st.text_input("Enter bundle name:")

# Section break
st.markdown("---")

# Button to trigger recommendation with styled button
if st.button("🔍 Recommend"):
    if uploaded_file is not None:
        # Load data and make recommendations
        model.load_data(uploaded_file)
        recommendations = model.get_users_to_activate_bundles(bundle_name, N=3)

        # Display the recommendations
        st.write("🎉 Recommendations:")
        st.write(recommendations)
    else:
        st.warning("Please upload a CSV file.")

# Section break
st.markdown("---")

# Button to trigger dataset analysis with styled button
if st.button("📊 Analyze Dataset"):
    if uploaded_file is not None:
        # Load data
        @st.cache(allow_output_mutation=True)
        def get_pyg_html(df: pd.DataFrame) -> str:
            html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
            return html
        
        @st.cache(allow_output_mutation=True)
        def get_df() -> pd.DataFrame:
            return pd.read_csv(uploaded_file)
        
        data = get_df()
        
        components.html(get_pyg_html(data), width=1300, height=1024, scrolling=True)
    else:
        st.warning("Please upload a CSV file for dataset analysis.")

# Section break
st.markdown("---")

# Button to load model and interact with styled button
if st.button("🤖 Load Model and Talk"):
    # Load the pre-trained model object
    with open("sketch_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    if uploaded_file is not None:
        # Read the contents of the file
        file_contents = uploaded_file.read()

        # Use BytesIO to create a file-like object from the file contents
        file_buffer = BytesIO(file_contents)

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_buffer)

        quest = st.text_input("Ask a question from the Data set!")

        # Use the pre-trained model object
        answer = model.ask(df, quest)

        # Display the answer
        st.write("🗣️ Answer:")
        st.write(answer)

    else:
        st.warning("Please upload a CSV file for dataset analysis")

st.markdown("---")

# Predicting customers who are ending their subscription
st.write('🕒 Predicting customers who are ending their subscription in the specified time interval')
time_interval = st.number_input("Input the max interval number starting today.")

# Dictionary to map bundle types to their respective values
bundle_type_mapping = {"Monthly": 30, "bi-Weekly": 14, "Weekly": 7, "bi-Monthly": 60 }

# Radio button for selecting bundle type
day = bundle_type_mapping[st.radio("Select Bundle Type", list(bundle_type_mapping.keys()))]

# Button to predict subscription validity with styled button
if st.button("🔮 Predict Subscription Validity"):
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Changing the activation data type
        data["ACTIVATION_DATE"] = pd.to_datetime(data["ACTIVATION_DATE"])
        
        # Filter customers ending subscription based on the specified time interval
        today = pd.to_datetime('today')
        end_date = pd.to_datetime('today') + pd.DateOffset(days=time_interval)
        data["DEACTIVATION_DATE"] = pd.to_datetime(data["ACTIVATION_DATE"]) + pd.DateOffset(days=day)
        
        # Assuming 'ACTIVATION_DATE' is the column containing the activation date
        ending_customers = data[
            (data['BUNDLE_NAME'] == bundle_name) & 
            (pd.to_datetime(data["ACTIVATION_DATE"]) + pd.DateOffset(days=30) > today) & 
            (pd.to_datetime(data["ACTIVATION_DATE"]) < end_date)
        ]

                # Display the customers
        st.write(f"📅 Customers Ending Subscription in the Next {time_interval} Days:")
        st.write(ending_customers[["SUBSCRIPTION_ID", "BUNDLE_NAME", "ACTIVATION_DATE", "DEACTIVATION_DATE"]]) 
    else:
        st.warning("Please upload a CSV file.")

st.markdown("---")

# Section for additional visualizations or analysis
# Add your custom visualizations or analysis here

# Custom styling with CSS
st.markdown(
    """
    <style>
        /* Add your custom CSS styling here */
        body {
            color: #2a2a2a;
            background-color: #065535;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #F0F8FF;
        }
        .stButton {
            color: #FFD700;
            background-color: #dddddd ;
            display: flex;
            justify-content: center;
        }
        .button-container {
            display: flex;
            justify-content: center;
        }
        .stCheckbox {
            color: #1f487e;
        }
        .stRadio {
            color: #1f487e;
        }
        .stTextInput {
            color: #1f487e;
        }
    </style>
    # Button to trigger recommendation
    <div class="button-container">
        <button class="stButton">Recommend</button>
    </div>
    """,
    unsafe_allow_html=True
    
)
