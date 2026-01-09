import torch
import torch.nn as nn

class CustomerSegmentNet(nn.Module):
    """
    Multi-class Classification Network for Customer Segment Prediction.
    Architecture: Dense MLP with Batch Normalization and Dropout for 
    robustness against small-to-medium dataset variance.
    """
    def __init__(self, input_dim, num_classes):
        super(CustomerSegmentNet, self).__init__()
        
        # Architecture Design
        self.network = nn.Sequential(
            # First Hidden Layer
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting on specific demographic noise
            
            # Second Hidden Layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Output Layer: Number of classes defined by K-Means clusters
            nn.Linear(32, num_classes) 
        )

    def forward(self, x):
        # We return raw logits. CrossEntropyLoss will handle the Softmax internally
        return self.network(x)