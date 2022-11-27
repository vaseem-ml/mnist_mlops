
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import numpy as np
from get_data import read_params
import argparse
import os



def load_data(config_path):
    config = read_params(config_path)
    processed_direcotory = config["data_source"]["processed"]

    BATCH_SIZE=64
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        processed_direcotory,
        validation_split=0.2,
        subset="training",
        seed=1337,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        processed_direcotory,
        validation_split=0.2,
        subset="validation",
    #     label_mode='categorical',
        seed=1337,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )


    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = load_data(config_path=parsed_args.config)
