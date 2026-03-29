import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import kagglehub
from kagglehub import KaggleDatasetAdapter


######################### Data #########################

class InsuranceDataset:
    DATASET_ID = "mirichoi0218/insurance"
    FILE_PATH   = "insurance.csv"
    FEATURES    = ["age", "bmi", "children"]
    TARGET      = "charges"

    def __init__(self):
        self.df: pd.DataFrame = self._load()

    def _load(self) -> pd.DataFrame:
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            self.DATASET_ID,
            self.FILE_PATH,
        )

    def describe(self) -> None:
        print(self.df.describe())

    def get_tensors(
        self,
        features: list[str] | None = None,
        target: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame | pd.Series, pd.Series]:
        """Return (X_tensor, y_tensor, X, y) for the requested columns."""
        features = features or self.FEATURES
        target   = target   or self.TARGET

        X = self.df[features]
        y = self.df[target]

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        # single-feature models expect shape (N, 1)
        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(1)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        return X_tensor, y_tensor, X, y


######################### Model #########################

class LinearModel:

    def __init__(self, n_features: int = 3):
        self.net = nn.Sequential(nn.Linear(n_features, 1))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()

    @property
    def weights(self) -> list[float]:
        return self.net[0].weight.tolist()

    @property
    def bias(self) -> float:
        return self.net[0].bias.item()

    def summary(self) -> None:
        print(f"Weights   : {self.weights}")
        print(f"Intercept : {self.bias:.2f}")


########################## Trainer #########################

class Trainer:
    def __init__(
        self,
        model: LinearModel,
        lr: float = 1e-5,
        loss_fn: nn.Module | None = None,
    ):
        self.model    = model
        self.loss_fn  = loss_fn or nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.loss_history: list[float] = []

    def train(
        self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        epochs: int = 200,
        log_every: int = 10,
    ) -> None:
        self.loss_history.clear()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss   = self.loss_fn(y_pred, y_tensor)
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())

            if epoch % log_every == 0:
                print(f"Epoch {epoch:>4}  |  Loss: {loss.item():.2f}")

        print(f"\nFinal loss: {self.loss_history[-1]:.2f}")


########################## Visualiser #########################

class Visualizer:

    def __init__(self, dataset: InsuranceDataset):
        self.dataset = dataset

    ########################## exploratory

    def plot_feature_vs_target(self, target: str = "charges") -> None:
        features = ["age", "bmi", "children", "charges"]
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))

        for ax, feature in zip(axes, features):
            ax.scatter(self.dataset.df[feature], self.dataset.df[target], alpha=0.4)
            ax.set_xlabel(feature)
            ax.set_ylabel(target)

        plt.tight_layout()
        plt.show()

    ########################## post-training

    def plot_single_feature_regression(
        self,
        X: pd.Series,
        y: pd.Series,
        model: LinearModel,
        X_tensor: torch.Tensor,
        loss_history: list[float],
    ) -> None:
        y_pred = model(X_tensor).detach().numpy().flatten()

        fig, (loss_ax, reg_ax) = plt.subplots(1, 2, figsize=(12, 4))

        loss_ax.plot(loss_history)
        loss_ax.set_title("Loss history")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("MSE")

        sorted_idx    = X.argsort()
        X_sorted      = X.values[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]

        reg_ax.scatter(X, y, alpha=0.4)
        reg_ax.plot(X_sorted, y_pred_sorted, color="red", label="regression")
        reg_ax.set_xlabel(X.name)
        reg_ax.set_ylabel("charges")
        reg_ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_multi_feature_surfaces(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: LinearModel,
        features: list[str] | None = None,
    ) -> None:
        features = features or InsuranceDataset.FEATURES
        pairs    = [(features[i], features[j]) for i in range(len(features))
                                                for j in range(i + 1, len(features))]

        fig = plt.figure(figsize=(18, 5))
        model.eval()

        for idx, (feat1, feat2) in enumerate(pairs, start=1):
            ax = fig.add_subplot(1, len(pairs), idx, projection="3d")
            ax.scatter(X[feat1], X[feat2], y, alpha=0.2)

            f1_range = np.linspace(X[feat1].min(), X[feat1].max(), 20)
            f2_range = np.linspace(X[feat2].min(), X[feat2].max(), 20)
            f1_grid, f2_grid = np.meshgrid(f1_range, f2_range)

            third      = [f for f in features if f not in (feat1, feat2)][0]
            third_mean = X[third].mean()

            grid_input = np.column_stack([
                {feat1: f1_grid.ravel(), feat2: f2_grid.ravel(),
                 third: np.full(f1_grid.size, third_mean)}[f]
                for f in features
            ])

            with torch.no_grad():
                z_grid = model(
                    torch.tensor(grid_input, dtype=torch.float32)
                ).numpy().reshape(f1_grid.shape)

            ax.plot_surface(f1_grid, f2_grid, z_grid, alpha=0.4, color="red")
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.set_zlabel("charges")
            ax.set_title(f"{feat1} vs {feat2}")

        plt.tight_layout()
        plt.show()


######################### Pipeline #########################

class InsurancePipeline:

    def __init__(
        self,
        features: list[str] | None = None,
        lr: float = 1e-5,
        epochs: int = 200,
    ):
        self.features = features or InsuranceDataset.FEATURES
        self.lr       = lr
        self.epochs   = epochs

        self.data      = InsuranceDataset()
        self.model     = LinearModel(n_features=len(self.features))
        self.trainer   = Trainer(self.model, lr=self.lr)
        self.visualizer = Visualizer(self.data)

    def run(self) -> None:
        # 1. Explore
        self.data.describe()

        # 2. Prepare tensors
        X_tensor, y_tensor, X, y = self.data.get_tensors(self.features)

        # 3. Train
        self.trainer.train(X_tensor, y_tensor, epochs=self.epochs)
        self.model.summary()

        # 4. Visualise
        if len(self.features) == 1:
            self.visualizer.plot_single_feature_regression(
                X.squeeze(), y, self.model, X_tensor, self.trainer.loss_history
            )
        else:
            self.visualizer.plot_multi_feature_surfaces(X, y, self.model, self.features)


########################## Entry point #########################
if __name__ == "__main__":
    pipeline = InsurancePipeline(
        features=InsuranceDataset.FEATURES,   # swap for ["age"] for single-input
        lr=1e-5,
        epochs=200,
    )
    pipeline.run()
