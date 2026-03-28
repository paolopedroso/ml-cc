# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import itertools
import mpl_toolkits.mplot3d  # noqa: F401 — registers projection='3d'
from load_data import load_data

def get_tensors(X_input, y_input):
    # z scaling
    X_normalized = (X_input - X_input.mean()) / X_input.std()
    return torch.tensor(X_normalized.values, dtype=torch.float32), torch.tensor(y_input.values, dtype=torch.float32)

def get_model(num_inputs, num_outputs):
    return nn.Sequential(nn.Linear(num_inputs, num_outputs))

def plot_final_data(data, features, label, X, y_pred):
    y_pred_np = y_pred.detach().numpy().flatten()

    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))

    for i, (ax, column) in enumerate(zip(axes, features)):
        sorted_idx = X[:, i].argsort()          # sort by this feature's column
        x_sorted = data[column].values[sorted_idx]
        y_sorted = y_pred_np[sorted_idx]

        ax.scatter(data[column], data[label], alpha=0.4)
        ax.plot(x_sorted, y_sorted, color="red", label="regression")
        ax.set_xlabel(column)
        ax.set_ylabel(label)
    plt.tight_layout()

    if not os.path.exists("./plot"):
        os.mkdir("plot")
    
    plt.savefig("./plot/final_figs.png", format="png")


def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    if not os.path.exists("./plot"):
        os.mkdir("plot")
    plt.savefig("./plot/loss.png", format="png")

if __name__ == "__main__":
    epochs = 1000

    # Set the path to the file you'd like to load
    file_path = "bestsellers with categories.csv"

    label = 'Price'
    columns = ['User Rating', 'Reviews', 'Year']

    X_features, y_labels, processed_dataset = load_data(columns, label, file_path)
    X_tensors, y_tensors = get_tensors(X_features, y_labels)

    # init model
    model = get_model(3, 1) # single three input layer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    # train model
    loss_history = []
    for iter in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensors)
        loss = loss_fn(y_pred.squeeze(), y_tensors)
        loss.backward() # backpropagation
        optimizer.step() # update weights

        if iter % 10 == 0:
            print(f"Iter: {iter} Loss: {loss}")

        loss_history.append(loss.detach().numpy())

    plot_loss(loss_history)
    plot_final_data(processed_dataset, columns, label, X_tensors, y_pred)
    
    print("Weights", model[0].weight.tolist())
    print("Bias", model[0].bias.tolist())    