import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

class CustomerDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def get_processed_loaders(file_path='segmented_customers.csv', batch_size=16):
    """
    Enhanced Data Processor with Feature Engineering for higher accuracy.
    """
    df = pd.read_csv(file_path)
    
    # --- 1. Feature Engineering (The Interpretation Layer) ---
    # Interaction: Value Density (Spend per Purchase)
    df['ValueDensity'] = df['TotalSpent'] / (df['NumPurchases'] + 1)
    
    # Temporal: Is Active Flag (Purchased in last 6 months)
    df['IsActive'] = (df['LastPurchaseDays'] < 180).astype(int)
    
    # Engagement: Click-to-Loyalty Ratio
    df['EngagementIndex'] = df['EmailClicks'] * df['LoyaltyScore']
    
    # --- 2. Feature Selection ---
    # We add these engineered features to our predictive set
    feature_cols = ['Age', 'Gender', 'Location', 'LoyaltyScore', 
                    'ValueDensity', 'IsActive', 'EngagementIndex']
    
    # --- 3. Categorical Encoding ---
    encoders = {}
    for col in ['Gender', 'Location']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # --- 4. Numerical Scaling ---
    # Scale all numerical features to help gradients converge faster
    num_cols = ['Age', 'LoyaltyScore', 'ValueDensity', 'EngagementIndex']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Save assets for App/Inference
    assets = {
        'encoders': encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'num_cols': num_cols
    }
    joblib.dump(assets, 'predictor_assets.joblib')
    
    # --- 5. Data Splitting ---
    X = df[feature_cols].values
    y = df['Segment'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(CustomerDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CustomerDataset(X_val, y_val), batch_size=batch_size)
    
    return train_loader, val_loader, len(feature_cols), len(np.unique(y))

if __name__ == "__main__":
    t_loader, v_loader, in_d, out_d = get_processed_loaders()
    print(f"Enhanced Processor Ready.")
    print(f"New Input Dimension: {in_d} (Increased from 4 to 7)")
    print(f"Target Segments: {out_d}")