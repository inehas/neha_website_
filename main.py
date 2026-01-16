import streamlit as st
from groq import Groq
import pandas as pd
import joblib

# 1. Load the model (Fixed Path)
model = joblib.load('models/xgboost_model.pkl')

# 2. Define the AI Function
def get_llm_explanation(customer_data, prediction_prob):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    prompt = f"""
    A bank customer has the following profile:
    - Age: {customer_data['Age']}
    - Balance: {customer_data['Balance']}
    - Credit Score: {customer_data['CreditScore']}
    - Number of Products: {customer_data['NumOfProducts']}
    - Geography: {customer_data['Geography']}
    
    The ML model predicts a {prediction_prob:.1%} probability of this customer churning.
    Briefly explain why this customer might be at risk or safe in 2-3 sentences.
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content

# 3. UI - Create the "Ingredients" (Inputs)
st.title("🏦 Bank Churn Predictor")
estimated_salary = st.text_input("Estimated Salary")

credit_score = st.number_input("Credit Score", 300, 850, 600)
age = st.number_input("Age", 18, 100, 40)
balance = st.number_input("Balance", 0.0, 200000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
# Added these because your model likely needs them
tenure = st.slider("Tenure", 0, 10, 5)
has_cr_card = st.checkbox("Has Credit Card")
is_active = st.checkbox("Is Active Member")
if st.button("Predict"):
    # 1. Map Geography
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}

    # 2. Create the dictionary with EXPLICIT type conversion
    # We use float() and int() to satisfy XGBoost's requirements
    user_input_dict = {
        'CreditScore': int(credit_score),
        'Geography': geo_map[geography],
        'Gender': 0, # Hardcoded as we discussed
        'Age': int(age),
        'Tenure': int(tenure),
        'Balance': float(balance),
        'NumOfProducts': int(num_products),
        'HasCrCard': int(has_cr_card),
        'IsActiveMember': int(is_active),
        'EstimatedSalary': float(estimated_salary) # Forced to float!
    }

    # 3. Convert to DataFrame
    input_df = pd.DataFrame([user_input_dict])
    
    # 4. Ensure Column Order
    column_order = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
        'Balance', 'NumOfProducts', 'HasCrCard', 
        'IsActiveMember', 'EstimatedSalary'
    ]
    input_df = input_df[column_order]

    # 5. Predict
    try:
        # We also ensure the DataFrame itself is float32 to be extra safe
        input_df = input_df.astype(float) 
        
        prob = model.predict_proba(input_df)[0][1]
        
        # 6. AI Explanation
        explanation = get_llm_explanation(user_input_dict, prob)
        
        st.write(f"### Churn Probability: {prob:.1%}")
        st.subheader("🤖 AI Insights")
        st.info(explanation)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")