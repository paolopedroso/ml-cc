# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

#
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Set the path to the file you'd like to load
file_path = "Salary_dataset.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "abhishek14398/salary-dataset-simple-linear-regression",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

# observe features
# print("First 5 records:\n", df.head())

salary_dataset = df[[
    'YearsExperience',
    'Salary'
]]

print(salary_dataset.describe())

print(salary_dataset.shape)

# init
# plt.scatter(salary_dataset['YearsExperience'], salary_dataset['Salary'])
# plt.xlabel('YearsExp')
# plt.ylabel('Salary')
# plt.show()

X = salary_dataset['YearsExperience']
# salary as our labels
y = salary_dataset['Salary']

# create then compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# input_optimizer="sgd"
# input_optimizer="adam"
input_optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=input_optimizer, loss="mean_squared_error")

history = model.fit(X, y, epochs=100, batch_size=5)
loss_history = history.history["loss"]
weight, bias = model.layers[0].get_weights()

print(f"Slope: {weight[0][0]:.2f}")
print(f"Intercept: {bias[0]:.2f}")

final_loss = history.history['loss'][-1]

print(f"Final loss: {final_loss:.2f}")

# predictions
y_pred = model.predict(X)

figs, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))

loss_plt = axes[0]
linreg_plt = axes[1]

loss_plt.plot(loss_history)
loss_plt.set_title("Loss")
loss_plt.set_xlabel("Loss over time")
loss_plt.set_ylabel("MSE")

linreg_plt.scatter(X, y, alpha=0.4, label="data")
linreg_plt.plot(X, y_pred, color="red", label="pred")
linreg_plt.set_xlabel('YearsExp')
linreg_plt.set_ylabel('Salary')
linreg_plt.legend()

plt.show()