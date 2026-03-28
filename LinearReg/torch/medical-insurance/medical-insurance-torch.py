# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, TensorDataset

# Set the path to the file you'd like to load
file_path = "insurance.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "mirichoi0218/insurance",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

def describe_data():
  return print(df.describe())

def get_weights(model):
  return model[0].weight.tolist(), model[0].bias.item()

def initial_data_plot(dataset, label = 'charges'):
  features = ['age', 'bmi', 'children', 'charges']

  fig, axes = plt.subplots(1, 4, figsize=(17, 4))

  for ax, feature in zip(axes, features):
      ax.scatter(dataset[feature], dataset[label], alpha=0.4)
      ax.set_xlabel(feature)
      ax.set_ylabel(label)

  plt.tight_layout()
  plt.show()

def train_model(model, optimizer, loss_fn, epochs=100):
  global loss_history
  global y_pred

  loss_history = []

  for iter in range(epochs):
    curr_loss = 0.0

    optimizer.zero_grad() # clear old gradients
    y_pred = model(X_tensor) # forward pass
    loss = loss_fn(y_pred, y_tensor) # compute loss
    loss.backward() # backward propagation (compute gradients)
    optimizer.step() # updates weights
    
    # loss.item() returns actual loss value
    curr_loss += loss.item()
    
    if iter % 10 == 0:
      print(f"Iteration: {iter} Current Loss: {curr_loss}")

    loss_history.append(loss.item())

  # return trained model
  return model

def single_input_layer():
  X, y = insurance_dataset['age'], insurance_dataset[['charges']]

  # add extra dimension, model accepts 2D and above dimensions
  X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
  y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
  
  # single io layer, dense layer
  model = nn.Sequential(
      nn.Linear(1, 1)
  )

  return model, X_tensor, y_tensor

def multi_input_layer():
  X, y = insurance_dataset[['age', 'bmi', 'children']], insurance_dataset['charges']

  # add extra dimension, model accepts 2D and above dimensions
  X_tensor = torch.tensor(X.values, dtype=torch.float32)
  y_tensor = torch.tensor(y.values, dtype=torch.float32)
  
  # 3 input layer, dense layer
  model = nn.Sequential(
      nn.Linear(3, 1)
  )

  return model, X_tensor, y_tensor

def final_data_plot_single_layer():
  fig, axes = plt.subplots(ncols=2, nrows=1)

  loss_plot = axes[0]
  linreg_plot = axes[1]

  loss_plot.plot(loss_history)
  loss_plot.set_title("Loss history")
  loss_plot.set_xlabel("Epoch")
  loss_plot.set_ylabel("MSE")

  sorted_idx = X.argsort()
  X_sorted = X.values[sorted_idx]
  y_pred_sorted = y_pred.detach().numpy().flatten()[sorted_idx]

  linreg_plot.scatter(X, y, alpha=0.4)
  linreg_plot.plot(X_sorted, y_pred_sorted, color="red", label="regression")
  linreg_plot.set_xlabel("age")
  linreg_plot.set_ylabel("charges")
  linreg_plot.legend()
  plt.show()

def final_data_plot_all_combinations():
    combinations = [('age', 'bmi'), ('age', 'children'), ('bmi', 'children')]
    fixed_feature = {'age': 'children', 'bmi': 'children', 'children': 'age'}

    fig = plt.figure(figsize=(18, 5))

    for i, (feat1, feat2) in enumerate(combinations):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.scatter(X[feat1], X[feat2], y, alpha=0.2)

        f1_range = np.linspace(X[feat1].min(), X[feat1].max(), 20)
        f2_range = np.linspace(X[feat2].min(), X[feat2].max(), 20)
        f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)

        # fix the third feature to its mean
        features = ['age', 'bmi', 'children']
        third = [f for f in features if f != feat1 and f != feat2][0]
        third_mean = X[third].mean()

        # build grid input in correct feature order
        grid_dict = {feat1: f1_grid.ravel(), feat2: f2_grid.ravel(), third: np.full(f1_grid.size, third_mean)}
        grid_input = np.column_stack([grid_dict[f] for f in features])

        grid_tensor = torch.tensor(grid_input, dtype=torch.float32)
        with torch.no_grad():
            z_grid = trained_model(grid_tensor).numpy().reshape(f1_grid.shape)

        ax.plot_surface(f1_grid, f2_grid, z_grid, alpha=0.4, color='red')
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_zlabel('charges')
        ax.set_title(f'{feat1} vs {feat2}')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
  input_label = 'charges'

  describe_data()

  insurance_dataset = df[['age','bmi','children','charges']]

  # notice relationship between age and charges
  # initial_data_plot(insurance_dataset, label=input_label)

  # model, X_tensor, y_tensor = single_input_layer()
  model, X_tensor, y_tensor = multi_input_layer()

  tdataset = TensorDataset(X_tensor, y_tensor)
  
  # stochastic optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
  # loss fn
  loss_fn = nn.MSELoss()

  trained_model = train_model(model, optimizer, loss_fn, epochs=200)

  weight, bias = get_weights(trained_model)

  print(f"Weights: {weight}")
  print(f"Intercept: {bias:.2f}")
  print(f"Final loss: {loss_history[-1]:.2f}")

  trained_model.eval()

  # final_data_plot_single_layer()
  final_data_plot_all_combinations()