
import torch
import torch.nn as nn
import pandas as pd
import os
import joblib
import numpy as np

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

class MLService:
    def __init__(self):
        # Load model
        model_path = os.path.join('models', 'chart_classifier.pth')
        scaler_path = os.path.join('models', 'scaler.save')
        encoder_path = os.path.join('models', 'label_encoder.save')

        self.model = PlotTypeClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

    def extract_csv_metadata(self, df):
        num_rows = len(df)
        num_cols = df.shape[1]
        num_numeric = len(df.select_dtypes(include=['number']).columns)
        num_categorical = num_cols - num_numeric
        avg_unique = int(np.mean([df[col].nunique() for col in df.columns[:min(5, num_cols)]]))

        return [[num_rows, num_cols, num_numeric, num_categorical, avg_unique]]

    def predict_plot_type(self, df):
        metadata = self.extract_csv_metadata(df)
        metadata_scaled = self.scaler.transform(metadata)
        metadata_tensor = torch.tensor(metadata_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(metadata_tensor)
            pred_index = output.argmax(dim=1).item()
            chart_type = self.encoder.inverse_transform([pred_index])[0]
            return chart_type
