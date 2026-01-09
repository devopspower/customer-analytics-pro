import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
from model import CustomerSegmentNet

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Insight AI",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_advanced_assets():
    """Loads transformation metadata and the trained PyTorch model."""
    assets = joblib.load('predictor_assets.joblib')
    input_dim = len(assets['feature_cols'])
    num_classes = 4 
    
    model = CustomerSegmentNet(input_dim, num_classes)
    model.load_state_dict(torch.load('segment_predictor.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, assets

def main():
    st.title("📊 Customer Segmentation & Strategy Engine")
    st.markdown("#### *Leveraging Deep Learning for Targeted Marketing Actions*")

    try:
        model, assets = load_advanced_assets()
    except Exception as e:
        st.error(f"Asset mismatch: {e}. Please re-run main.py with the updated data_processor.")
        return

    # --- UI Layout: Sidebar for Inputs ---
    st.sidebar.header("🛠️ Customer Profile Input")
    with st.sidebar:
        st.subheader("Demographics")
        age = st.number_input("Age", 18, 100, 35)
        gender = st.selectbox("Gender", list(assets['encoders']['Gender'].classes_))
        location = st.selectbox("Location", list(assets['encoders']['Location'].classes_))
        
        st.subheader("Engagement & Spend")
        loyalty = st.slider("Loyalty Score (1-100)", 1, 100, 50)
        total_spent = st.number_input("Total Lifetime Spend ($)", 0.0, 50000.0, 1500.0)
        num_purchases = st.number_input("Total Number of Purchases", 1, 500, 10)
        days_since_last = st.number_input("Days Since Last Purchase", 0, 1000, 30)
        email_clicks = st.number_input("Email Clicks (Monthly)", 0, 100, 5)
        
        predict_btn = st.button("Generate Strategy", type="primary", use_container_width=True)

    if predict_btn:
        # --- 1. Real-Time Feature Engineering ---
        value_density = total_spent / (num_purchases + 1)
        is_active = 1 if days_since_last < 180 else 0
        engagement_index = email_clicks * loyalty
        
        # --- 2. Transform & Scale ---
        g_enc = assets['encoders']['Gender'].transform([gender])[0]
        l_enc = assets['encoders']['Location'].transform([location])[0]
        
        # Scale numericals: ['Age', 'LoyaltyScore', 'ValueDensity', 'EngagementIndex']
        scaled_nums = assets['scaler'].transform([[age, loyalty, value_density, engagement_index]])
        
        # --- 3. Prepare Tensor ---
        input_array = np.array([[
            scaled_nums[0][0], g_enc, l_enc, scaled_nums[0][1], 
            scaled_nums[0][2], is_active, scaled_nums[0][3]
        ]])
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # --- 4. Inference ---
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs).item()
            confidence = probs[0][prediction].item()

        # --- 5. Persona Mapping (Refined Labels) ---
        persona_map = {
            0: {
                "name": "Active Spender", 
                "behavior": "High-Engagement / Low-Spend", 
                "color": "#3498db", "tag": "Upsell"
            },
            1: {
                "name": "VIP Loyalist", 
                "behavior": "High-Value / Frequent", 
                "color": "#2ecc71", "tag": "Retain"
            },
            2: {
                "name": "New Explorer", 
                "behavior": "High-Ticket / Emerging", 
                "color": "#f39c12", "tag": "Nurture"
            },
            3: {
                "name": "At-Risk Buyer", 
                "behavior": "Lapsed / High-Potential", 
                "color": "#e74c3c", "tag": "Re-engage"
            }
        }
        
        res = persona_map[prediction]
        
        # --- 6. Results Display ---
        st.subheader("🎯 Prediction Analysis")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Predicted Persona", res['name'])
            st.caption(f"**Behavior:** {res['behavior']}")
            
        with c2:
            st.metric("Model Confidence", f"{confidence:.2%}")
            st.progress(confidence)
            
        with c3:
            st.metric("Strategic Priority", res['tag'])
            st.info(f"**Action:** Apply {res['tag']} campaign logic.")

        st.divider()
        
        # Probability Distribution Visual
        st.write("#### Confidence Distribution across All Segments")
        chart_data = pd.DataFrame({
            "Segment": [persona_map[i]['name'] for i in range(4)],
            "Probability": probs[0].tolist()
        })
        st.bar_chart(chart_data.set_index("Segment"))

if __name__ == "__main__":
    main()