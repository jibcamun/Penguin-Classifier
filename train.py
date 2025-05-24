import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv("data/penguins_size.csv")

# Encode categorical columns
categorical_cols = ["studyName", "Sample Number", "Individual ID", "Date Egg", 
                    "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Comments", 
                    "Species", "Region", "Island", "Stage", "Clutch Completion", 
                    "Sex"]
for col in categorical_cols:
    data[col] = pd.factorize(data[col])[0]

# Drop unnecessary columns and handle missing values
data = data.drop(columns=["studyName", "Sample Number", "Individual ID", "Date Egg", 
                          "Delta 15 N (o/oo)", "Delta 13 C (o/oo)", "Comments"])
data = data.dropna()

# Split features and labels
X = data.drop(columns=["Species"]).values
y = data["Species"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

# Create DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class PenguinClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, len(set(y)))
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize model, optimizer, and loss function
model = PenguinClassificationModel().to("cpu")
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
if __name__ == "__main__":
    for epoch in range(1000):
        for batch in dataloader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to("cpu"), y_batch.to("cpu")
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Evaluate model
    with torch.no_grad():
        y_eval = model(X_test)
        eval_loss = criterion(y_eval, y_test)
    
    print(f"Final Loss: {eval_loss.item()}")

    # Save model
    with open("model.pt", 'wb') as f:
        save(model.state_dict(), f)
