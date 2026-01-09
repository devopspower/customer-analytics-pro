import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def analyze_and_segment(file_path, output_labeled_path='segmented_customers.csv'):
    """
    Stage 1: Unsupervised discovery of customer segments.
    Uses behavioral features to identify 'High-Value', 'At-Risk', and 'Engaged' groups.
    """
    # 1. Load behavioral data
    df = pd.read_csv(file_path)
    
    # Define behavioral features for clustering
    behav_cols = [
        'TotalSpent', 'NumPurchases', 'AvgOrderValue', 
        'LastPurchaseDays', 'EmailClicks', 'LoyaltyScore'
    ]
    
    # 2. Preprocessing
    # Scaling is critical for K-Means (Distance-based)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[behav_cols])
    
    # 3. Find Optimal K (Elbow Method)
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
    
    # Plotting the Elbow Method result for validation
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, wcss, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method for Optimal K')
    plt.savefig('elbow_plot.png')
    print("Elbow plot saved as 'elbow_plot.png'. Inspect to confirm optimal K.")

    # 4. Apply Final Clustering
    # Assuming 4 clusters based on initial data observation
    optimal_k = 4 
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['Segment'] = kmeans_final.fit_predict(scaled_data)
    
    # 5. Profiling: Summarize segments for the 'Interpretation Layer'
    profile = df.groupby('Segment')[behav_cols].mean()
    print("\n--- Customer Segment Profiles (Means) ---")
    print(profile)
    
    # 6. Persistence: Save assets for the supervised Phase 2
    df.to_csv(output_labeled_path, index=False)
    joblib.dump(scaler, 'segment_scaler.joblib')
    joblib.dump(kmeans_final, 'kmeans_model.joblib')
    
    print(f"\nSuccess: Labeled dataset saved to {output_labeled_path}")
    return df

if __name__ == "__main__":
    analyze_and_segment('data/customer_segmentation_data.csv')