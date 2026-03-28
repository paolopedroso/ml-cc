import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

import matplotlib.pyplot as plt

def plot_loaded_data(data, features, label):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))

    for ax, column in zip(axes, features):
      ax.scatter(data[column], data[label], alpha=0.4)
      ax.set_xlabel(column)
      ax.set_ylabel(label)
    plt.tight_layout()

    if not os.path.exists("./plot"):
        os.mkdir("plot")
    
    plt.savefig("./plot/all_figs.png", format="png")

def load_data(columns, label, file_path):
    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "sootersaalu/amazon-top-50-bestselling-books-2009-2019",
    file_path,
    )
    
    print(
        "Prepocessed Dataset\n",
        "-"*40
    )
    print(df.describe())

    dataset = df[columns + [label]]

    if not os.path.exists("./data"):
        os.mkdir("data")
    try:
        with open("data/bestsellers.csv","w") as f:
            f.write(dataset.to_csv(index=False))
    except:
        raise FileNotFoundError
    
    plot_loaded_data(dataset, columns, label)

    X_features, y_labels = dataset[columns], dataset[label] 
    return X_features, y_labels, dataset
