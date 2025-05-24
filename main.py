import torch
import torch.nn as nn

class PenguinClassificationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, output_size)
        )

    def forward(self, x):
        return self.model(x)
    
model = PenguinClassificationModel(9, 3).to("cpu")
model.load_state_dict(torch.load("model.pt"))
model.eval()

sample = [50.0, 18.0, 195.0, 3750.0, 3700.0, 1, 1, 2, 1]
X_single = torch.FloatTensor(sample).unsqueeze(0)
with torch.no_grad():
    prediction = model(X_single)
    predicted_class = torch.argmax(prediction, dim=1).item()

print(f"Predicted Class: {predicted_class}")