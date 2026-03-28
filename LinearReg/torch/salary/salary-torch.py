# pip install kagglehub[pandas-datasets] torch matplotlib

import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

file_path = "Salary_dataset.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "abhishek14398/salary-dataset-simple-linear-regression",
    file_path,
)

salary_dataset = df[['YearsExperience', 'Salary']]
print(salary_dataset.describe())
print(salary_dataset.shape)

X = salary_dataset['YearsExperience'].values
y = salary_dataset['Salary'].values

# Convert to PyTorch tensors
# TF accepted raw numpy/pandas PyTorch needs explicit tensors
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape [N, 1]
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape [N, 1]

# DataLoader (equivalent to batch_size=5 in model.fit)
# TF handled batching internally PyTorch uses DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Model (equivalent to tf.keras.Sequential + Dense(1))
# nn.Linear(in, out) = Dense layer handles weight + bias automatically
model = nn.Sequential(
    nn.Linear(1, 1)
)

# Optimizer (equivalent to tf.keras.optimizers.SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Loss function (equivalent to loss="mean_squared_error")
loss_fn = nn.MSELoss()

# Training loop (equivalent to model.fit(epochs=100))
# This is the biggest difference PyTorch requires a manual loop
# TF hides this; PyTorch exposes it giving you full control
loss_history = []

for epoch in range(100):
    epoch_loss = 0.0

    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()       # 1. clear old gradients (TF does this auto)
        y_pred = model(X_batch)     # 2. forward pass
        loss = loss_fn(y_pred, y_batch)  # 3. compute loss
        loss.backward()             # 4. backpropagation (compute gradients)
        optimizer.step()            # 5. update weights
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)

# Extract weights (equivalent to model.layers[0].get_weights())
weight = model[0].weight.item()
bias   = model[0].bias.item()

print(f"Slope:     {weight:.2f}")
print(f"Intercept: {bias:.2f}")
print(f"Final loss: {loss_history[-1]:.2f}")

# Predictions (equivalent to model.predict(X))
model.eval()                        # switch off dropout/batchnorm (good habit)
with torch.no_grad():               # disable gradient tracking for inference
    y_pred = model(X_tensor).numpy()

# Plotting (identical to before)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))

axes[0].plot(loss_history)
axes[0].set_title("Loss")
axes[0].set_xlabel("Loss over time")
axes[0].set_ylabel("MSE")

axes[1].scatter(X, y, alpha=0.4, label="data")
axes[1].plot(X, y_pred, color="red", label="pred")
axes[1].set_xlabel('YearsExp')
axes[1].set_ylabel('Salary')
axes[1].legend()

plt.show()
