
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Simulate metadata from CSV files
data = []
for _ in range(300):
    num_rows = int(torch.randint(5, 1000, (1,)).item())
    num_cols = int(torch.randint(1, 10, (1,)).item())
    num_numeric = int(torch.randint(0, num_cols, (1,)).item())
    num_categorical = num_cols - num_numeric
    avg_unique = int(torch.randint(2, min(num_rows, 20), (1,)).item())

    # Simulate a rule for chart type
    if num_categorical >= 1 and avg_unique <= 10:
        label = 'pie'
    elif num_numeric >= 1 and num_rows > 10:
        label = 'line'
    else:
        label = 'bar'

    data.append([num_rows, num_cols, num_numeric, num_categorical, avg_unique, label])

df = pd.DataFrame(data, columns=["num_rows", "num_cols", "num_numeric", "num_categorical", "avg_unique", "label"])

# Label encode chart type
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Features and labels
X = df[["num_rows", "num_cols", "num_numeric", "num_categorical", "avg_unique"]].values
y = df["label_encoded"].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define model
class PlotTypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.fc(x)

model = PlotTypeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluate accuracy
model.eval()
preds = model(X_test_tensor).argmax(dim=1)
accuracy = (preds == y_test_tensor).float().mean().item()
print(f"Test Accuracy: {accuracy:.2f}")

# Save model and encoders
torch.save(model.state_dict(), "chart_classifier.pth")
import joblib
joblib.dump(scaler, "scaler.save")
joblib.dump(le, "label_encoder.save")
