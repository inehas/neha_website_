import streamlit as st
from groq import Groq # Import the Groq library

# Initialize the Groq client using your secret key
# Streamlit automatically finds the key from .streamlit/secrets.toml
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_llm_explanation(customer_data, prediction_prob):
    # Construct a specific prompt using the features from your notebook
    prompt = f"""
    A bank customer has the following profile:
    - Age: {customer_data['Age']}
    - Balance: ${customer_data['Balance']:,}
    - Credit Score: {customer_data['CreditScore']}
    - Number of Products: {customer_data['NumOfProducts']}
    - Geography: {customer_data['Geography']}
    
    The ML model predicts a {prediction_prob:.1%} probability of this customer churning.
    Briefly explain why this customer might be at risk or safe in 2-3 sentences.
    Focus on financial and demographic indicators, not the model itself.
    """
    
    # Call the Groq API (using a fast model like Llama 3)
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile", # High-performance Groq model
    )
    
    return response.choices[0].message.content

# Inside your Streamlit "Predict" button logic:
if st.button("Predict"):
    # ... (prediction code from your model) ...
    explanation = get_llm_explanation(user_input_dict, prob)
    st.subheader("🤖 AI Insights")
    st.write(explanation)