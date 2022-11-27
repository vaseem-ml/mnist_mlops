import os
from get_data import read_params, get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import shutil



def getImageUsingPixels(indexNumber, df):
    imageData=[]
    row=[]
    for i in df.iloc[indexNumber]:
        if len(row)==28:
            imageData.append(row)
            row=[]
        row.append(i+1)
    return imageData


def process_data(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    if not os.path.isdir('data/processed'): os.mkdir('data/processed')
    for index, row in df.iterrows():
        image = getImageUsingPixels(index, df)
        data = Image.fromarray(np.array(image).astype(np.uint8))
        if not os.path.isdir(f'{config["data_source"]["processed"]}/{row["label"]}'): os.mkdir(f'{config["data_source"]["processed"]}/{row["label"]}')
        data.save(f'{config["data_source"]["processed"]}/{row["label"]}/{row["label"]}_{index}.png')


    ##### managing direcotyr for monitored
    classes = []

    for dir in os.listdir(f'{config["data_source"]["monitored"]}'):
        if len(os.listdir(os.path.join(f'{config["data_source"]["monitored"]}', dir))) > 200:
            classes.append(dir)

    if os.path.isdir(f'{config["data_source"]["processed"]}'): shutil.rmtree(f'{config["data_source"]["processed"]}')
    os.mkdir(config["data_source"]["processed"])

    for class_ in classes:
        os.system(f'cp -r {os.path.join(config["data_source"]["monitored"], class_)} {config["data_source"]["processed"]}/')




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    process_data(config_path=parsed_args.config)    
