import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set page config
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

# Load the model or train if not available
@st.cache_resource
def load_model():
    model = joblib.load('loan_approval_model.pkl')
    return model
        

# Load or train the model
model = load_model()

# Title and description
st.title("Loan Approval Prediction System")
st.write("This application predicts whether a loan application will be approved based on applicant information.")

# Create tabs
tab1, tab2 = st.tabs(["Prediction", "About"])

with tab1:
    st.header("Enter Applicant Details")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        cibil_score = st.slider("CIBIL Score", 300, 900, 600, help="Credit score ranging from 300-900")
        income = st.number_input("Annual Income (₹)", 200000, 10000000, 5000000, step=100000, 
                                help="Annual income in Rupees")
        loan_amount = st.number_input("Loan Amount (₹)", 300000, 40000000, 15000000, step=100000, 
                                    help="Requested loan amount in Rupees")
        loan_term = st.slider("Loan Term (years)", 2, 20, 10, help="Duration of loan in years")
        dependents = st.slider("Number of Dependents", 0, 5, 2, help="Number of people dependent on the applicant")
    
    with col2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        residential_assets = st.number_input("Residential Assets Value (₹)", 0, 30000000, 5000000, step=100000, 
                                            help="Value of residential properties owned")
        commercial_assets = st.number_input("Commercial Assets Value (₹)", 0, 20000000, 3000000, step=100000, 
                                          help="Value of commercial properties owned")
        luxury_assets = st.number_input("Luxury Assets Value (₹)", 300000, 40000000, 10000000, step=100000, 
                                      help="Value of luxury items owned")
        bank_assets = st.number_input("Bank Assets Value (₹)", 0, 15000000, 4000000, step=100000, 
                                    help="Value of assets in bank accounts")
    
    # Prediction button
    if st.button("Predict Loan Approval"):
        # Create input dataframe
        input_data = pd.DataFrame({
            ' no_of_dependents': [dependents],
            ' education': [education],
            ' self_employed': [self_employed],
            ' income_annum': [income],
            ' loan_amount': [loan_amount],
            ' loan_term': [loan_term],
            ' cibil_score': [cibil_score],
            ' residential_assets_value': [residential_assets],
            ' commercial_assets_value': [commercial_assets],
            ' luxury_assets_value': [luxury_assets],
            ' bank_asset_value': [bank_assets]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"Loan Approved with {probability[0][1]:.2%} confidence")
        else:
            st.error(f"Loan Rejected with {probability[0][0]:.2%} confidence")
        
        # Display feature importance
        st.subheader("Feature Importance Analysis")
        
        # Create feature importance visualization
        importance_data = {
            ' cibil_score': 0.81,
            ' loan_term': 0.07,
            ' loan_amount': 0.03,
            ' luxury_assets_value': 0.02,
            ' residential_assets_value': 0.02,
            ' income_annum': 0.01,
            ' commercial_assets_value': 0.01,
            ' bank_asset_value': 0.01,
            ' no_of_dependents': 0.01,
            ' self_employed': 0.005,
            ' education': 0.005
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(importance_data.values()), y=list(importance_data.keys()), ax=ax)
        ax.set_title('Feature Importance for Loan Approval')
        ax.set_xlabel('Importance')
        st.pyplot(fig)
        
        # Provide explanation
        st.subheader("Explanation")
        
        # CIBIL score explanation
        if cibil_score > 750:
            st.write("✅ High CIBIL score significantly increases approval chances.")
        elif cibil_score < 500:
            st.write("❌ Low CIBIL score significantly decreases approval chances.")
        
        # Loan amount to income ratio
        ratio = loan_amount / income
        if ratio > 5:
            st.write("❌ Loan amount to income ratio is high ({}), which may affect approval negatively.".format(round(ratio, 2)))
        else:
            st.write("✅ Loan amount to income ratio is reasonable ({}).".format(round(ratio, 2)))
        
        # Assets explanation
        total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
        if total_assets < loan_amount:
            st.write("❌ Total assets are less than the loan amount, which may affect approval negatively.")
        else:
            st.write("✅ Total assets adequately cover the loan amount.")
        
        # Loan term explanation
        if loan_term > 15:
            st.write("❌ Longer loan term may increase risk assessment.")
        else:
            st.write("✅ Moderate loan term is favorable for approval.")

with tab2:
    st.header("About This Application")
    st.write("""
    This loan approval prediction system uses machine learning to assess loan applications based on various factors.
    
    ### Model Information
    - **Algorithm**: Random Forest Classifier
    - **Accuracy**: 98.01%
    - **Key Features**: CIBIL score is the most important factor, followed by loan term and loan amount
    
    ### How It Works
    The model analyzes the applicant's financial profile, credit history, and other factors to determine the likelihood of loan approval. The prediction is based on patterns learned from thousands of previous loan applications.
    
    ### Data Privacy
    This application does not store any of the information entered. All predictions are made in real-time and no personal data is saved.
    """)
    
    # Display feature importance chart from the image
    st.subheader("Feature Importance Analysis")
    st.image("feature_importances.png", 
             caption="Feature Importance for Loan Approval Prediction")

# Footer
st.markdown("---")
st.markdown("© 2025 Loan Approval Prediction System | Created for ML Project by Pradipkumar Maganbhai Solanki and Akshay Vasudeva Rao.")
