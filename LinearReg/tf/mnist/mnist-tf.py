import os
import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Checking Tensorflow version... Tensorflow version:",tf.__version__)
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) =  mnist.load_data()
    x_test, x_train = x_test / 255.0, x_train / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()

    # loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # train on the first target
    loss_fn(y_train[:1], predictions).numpy()

    # configuration
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    # configure epochs
    model.fit(x_train, y_train, epochs=5)

    # finish
    model.evaluate(x_test,  y_test, verbose=2)

    probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])

    # plots

    probs = probability_model(x_test[:5]).numpy()

    # argmax finds the index of the highest score
    predicted_labels = probs.argmax(axis=1)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(x_test[i], cmap='gray')
        axes[i].set_title(f"Pred: {predicted_labels[i]}\nActual: {y_test[i]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()