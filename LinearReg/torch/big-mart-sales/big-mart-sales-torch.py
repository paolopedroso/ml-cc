# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split


from dataclasses import dataclass, field

########################## Dataprocessing #########################

@dataclass
class ProcessedData:
	X_dirty: pd.DataFrame
	y_dirty: pd.DataFrame
	X_clean: pd.DataFrame = field(default_factory=lambda: torch.tensor([]))
	y_clean: pd.DataFrame = field(default_factory=lambda: torch.tensor([]))

	def __post_init__(self):
		if not isinstance(self.X_dirty, (pd.DataFrame, pd.Series)):
			raise ValueError
		if not isinstance(self.y_dirty, (pd.DataFrame, pd.Series)):
			raise ValueError
		
		self.X_clean = self.X_dirty.fillna(self.X_dirty.median())
		self.y_clean = self.y_dirty.fillna(self.y_dirty.median())
	
	def get_dirty_data(self):
		return self.X_dirty, self.y_dirty
	
	def get_normalized_values(self):
		return self.get_zscaled_value(self.X_clean), self.get_log_value(self.y_clean)

	def get_clean_data(self):
		return self.X_clean, self.y_clean

	@staticmethod
	def get_zscaled_value(val: pd.DataFrame):
		return (val - val.mean()) / val.std()

	@staticmethod
	def get_log_value(val: pd.DataFrame):
		return np.log(val)

########################## Dataset #########################

class BigMartSalesDataset():
	file_path = "big_mart_sales.csv"
	dataset_id = "ailearner-researchlab/big-mart-sales-prediction-beginners-choice"

	def __init__(self, feature, label):
		self.feature, self.label = feature, label
		self.df = self._load()


	def _load(self):
		return kagglehub.load_dataset(
			KaggleDatasetAdapter.PANDAS,
			self.dataset_id,
			self.file_path
		)

	def describe(self) -> None:
		""" describe via pandas describe() method """
		return print(self.df.describe())
	
	def get_data(self) -> ProcessedData:
		""" fill in NaNs """
		X_dirty, y_dirty = self.df[self.feature], self.df[self.label]
		return ProcessedData(
			X_dirty=X_dirty, 
			y_dirty=y_dirty
		)

	def get_training_split(self, test_size=0.2):
		""" returns X y numerical data and tensors for training """
		
		data = self.get_data()
		X_clean, y_clean = data.get_clean_data()
		X_normalized, y_normalized = data.get_normalized_values()

		X_tensor = torch.tensor(X_normalized, dtype=torch.float32).reshape(-1,1)
		y_tensor = torch.tensor(y_normalized, dtype=torch.float32).reshape(-1,1)
		X_train, X_test, y_train, y_test = train_test_split(
			X_tensor, y_tensor,
			test_size=test_size,
			random_state=42
		)

		return X_train, X_test, y_train, y_test
########################## Visualiser #########################

class Visualize:
	def __init__(self, input_feature, input_label):
		self.input_feature = input_feature
		self.input_label = input_label

	def init_v_norm_plots(self, data: ProcessedData) -> None:
		X_dirty, y_dirty = data.get_dirty_data()
		X_normalized, y_normalized = data.get_normalized_values()

		init_plot, init_axes = plt.subplots(nrows=2, ncols=2, figsize = (15, 10))
		item_weight_plot = init_axes[0][0]
		item_weight_plot.hist(X_dirty)
		item_weight_plot.set_title(self.input_feature + " Before Normalization")
		item_weight_plot.set_xlabel(self.input_feature + " Normalized")
		item_weight_plot.set_ylabel("Frequency")	

		# normalization
		item_weight_plot_normalized = init_axes[0][1]
		item_weight_plot_normalized.hist(X_normalized)
		item_weight_plot_normalized.set_xlabel(self.input_feature + " Normalized")
		item_weight_plot_normalized.set_ylabel("Frequency")
		item_weight_plot_normalized.set_title(self.input_feature + " After Normalization")

		item_outlet_sales_plot = init_axes[1][0]
		item_outlet_sales_plot.hist(y_dirty)
		item_outlet_sales_plot.set_xlabel("Item Output Sales Before Normalization")
		item_outlet_sales_plot.set_ylabel("Frequency")
		item_outlet_sales_plot.set_title("Item Output Sales")	

		item_outlet_sales_plot_normalized = init_axes[1][1]
		item_outlet_sales_plot_normalized.hist(y_normalized)
		item_outlet_sales_plot_normalized.set_xlabel("Item Output Sales After Normalized")
		item_outlet_sales_plot_normalized.set_ylabel("Frequency")
		item_outlet_sales_plot_normalized.set_title("Item Output Sales")
		init_plot.savefig("./plot/distributions.png")
		return

	def feature_v_label_plots(self, data: ProcessedData) -> None:
		X_dirty, y_dirty = data.get_dirty_data()
		X_normalized, y_normalized = data.get_normalized_values()

		v_figs, v_axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

		outletvweight_plot = v_axes[0]
		outletvweight_plot.scatter(X_dirty, y_dirty)
		outletvweight_plot.set_xlabel("X features")
		outletvweight_plot.set_ylabel("y labels")
		outletvweight_plot.set_title("Output v " + self.input_feature + " Before")	

		outletvweight_plot = v_axes[1]
		outletvweight_plot.scatter(X_normalized, y_normalized)
		outletvweight_plot.set_xlabel("X features")
		outletvweight_plot.set_ylabel("y labels")
		outletvweight_plot.set_title("Output v " + self.input_feature + " After")
		v_figs.savefig("./plot/feature_vs_label.png")
		return
	
	def final_plots(self,
			data: ProcessedData,
			model: nn.Linear,
			loss_history: list
		) -> None:
		X_normalized, y_normalized = data.get_normalized_values()
		X_tensor = torch.tensor(X_normalized.values, dtype=torch.float32).reshape(-1, 1)
		y_tensor = torch.tensor(y_normalized.values, dtype=torch.float32).reshape(-1, 1)
		with torch.no_grad():
			y_pred = model(X_tensor)
		loss_plot, loss_axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		loss_axes.plot(loss_history)
		loss_axes.set_xlabel("Epoch")
		loss_axes.set_ylabel("Loss")
		loss_axes.set_title("Training Loss")
		loss_plot.savefig("./plot/loss.png")

		final_plot, final_axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		sorted_idx = X_tensor.reshape(-1,).argsort()
		X_sorted = X_tensor[sorted_idx].detach().numpy().reshape(-1,)
		y_sorted = y_pred[sorted_idx].detach().numpy().reshape(-1,)
		final_axes.plot(X_sorted, y_sorted, color="red")
		final_axes.scatter(X_tensor, y_tensor)
		final_axes.set_xlabel(self.input_feature + " (normalized)")
		final_axes.set_ylabel(self.input_label + " (log)")
		final_axes.set_title("Regression Fit")
		final_plot.savefig("./plot/final_fit.png")
		return

########################## Model #########################

class LinearModel:
	def __init__(self):
		self.net = nn.Sequential(
			nn.Linear(1, 1)
		)

	def __call__(self, X: torch.tensor):
		return self.net(X)
	
	def parameters(self):
		return self.net.parameters()

	@property
	def weights(self):
		return self.net[0].weight.tolist()
	
	@property
	def bias(self):
		return self.net[0].weight.item()

########################## Trainer #########################

class Trainer:
	def __init__(self, lr=0.001):
		self.loss_history = []
		self.loss_fn = nn.MSELoss()
		self.model = LinearModel()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

	def train(self,
			X: torch.tensor,
			y_actual: torch.tensor,
			epochs: int
		) -> None:
		self.loss_history.clear()
		for epoch in range(epochs):
			self.optimizer.zero_grad() # clear gradients
			y_pred = self.model(X) # linear regression equation
			loss = self.loss_fn(y_pred, y_actual) # calculate mse loss
			loss.backward() # computes gradient
			self.optimizer.step() # update weights
			if epoch % 10 == 0:
				print(f"Epoch {epoch:>4}  |  Loss: {loss.item():.2f}")
			self.loss_history.append(loss.detach().item())

		return None

	def evaluate(self, 
			X_test: torch.Tensor, 
			y_test: torch.Tensor
		) -> float:
		self.model.net.eval()     
		with torch.no_grad(): 
			y_pred = self.model(X_test)
			loss = self.loss_fn(y_pred, y_test)
		print(f"Test Loss: {loss.item():.4f}")
		return loss.item()

class Pipeline:
	def __init__(self, feature, label, lr=0.001):
		self.data = BigMartSalesDataset(feature, label)
		self.viz = Visualize(feature, label)
		self.trainer = Trainer(lr=lr)

		self.feature = feature
		self.label = label

	def run(self):

		#
		self.data.describe()

		X_train, X_test, y_train, y_test = self.data.get_training_split()

		processed = self.data.get_data()

		self.viz.init_v_norm_plots(processed)
		self.viz.feature_v_label_plots(processed)

		self.trainer.train(X_train, y_train, epochs=300)

		print(f"[FINAL] Weight {self.trainer.model.weights} | {self.trainer.model.bias}")

		self.trainer.evaluate(X_test, y_test)

		self.viz.final_plots(processed, self.trainer.model, self.trainer.loss_history)

# entry point
if __name__ == "__main__":
	pipeline = Pipeline(
		'Item_MRP',
		'Item_Outlet_Sales',
		lr=0.005
	)
	pipeline.run()
