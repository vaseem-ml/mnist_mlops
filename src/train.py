import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.metrics import precision_score, f1_score, recall_score
import os
import json
from get_data import read_params
from load_data import load_data
import argparse





def eval_metrics(validation_dataset, model):

    yhat_probs = model.predict(validation_dataset, verbose=0)
    y_true = []
    for batch in validation_dataset:
        _, labels = batch
        for class_ in labels:
            y_true.append(class_.numpy())


    y_pred = yhat_probs.argmax(axis=1)
    precision = precision_score(y_true, y_pred,average='macro')
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")  
    return precision, f1, recall



def train(config_path):
    train_dataset, validation_dataset = load_data(config_path)
    config = read_params(config_path)
    classes = len(os.listdir(config["data_source"]["processed"]))


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(256, 256, 1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


    H = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config["estimators"]["epochs"]
    )


    (precision, f1, recall) = eval_metrics(validation_dataset, model)
    scores_file = config["reports"]["scores"]
    with open(scores_file, "w") as f:
            scores = {
                "precision": precision,
                "f1": f1,
                "recall": recall
            }
            json.dump(scores, f, indent=4)
    model.save('saved_models/new_model.h5')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = train(config_path=parsed_args.config)