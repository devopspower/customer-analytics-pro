import torch
import torch.nn as nn
import torch.optim as optim
from segmenter import analyze_and_segment
from data_processor import get_processed_loaders
from model import CustomerSegmentNet
import numpy as np

def train_pipeline():
    # 1. Execute Clustering to generate labels
    print("--- Stage 1: Unsupervised Customer Segmentation ---")
    analyze_and_segment('data/customer_segmentation_data.csv')
    
    # 2. Load processed data for the Neural Network
    print("\n--- Stage 2: Supervised Predictive Analytics ---")
    train_loader, val_loader, input_dim, num_classes = get_processed_loaders()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomerSegmentNet(input_dim, num_classes).to(device)
    
    # CrossEntropyLoss is ideal for Multi-class (Segments 0, 1, 2, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    epochs = 40
    best_acc = 0.0
    
    print(f"Training on: {device} | Input Features: {input_dim} | Segments: {num_classes}")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        # 3. Evaluation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = np.mean(train_losses)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Accuracy: {accuracy:.2f}%")
            
        # Save the best performing model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'segment_predictor.pth')

    print("\n--- Pipeline Execution Complete ---")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("Predictive model saved as: segment_predictor.pth")

if __name__ == "__main__":
    train_pipeline()