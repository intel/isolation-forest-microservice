# main.py: Entrypoint for service application. Contains functions related to built-in datasets, custom dataset loading, data plotting, and the actual microservice API.

# Copyright (c) 2025 Intel Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
import argparse
import tomllib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import time
import h5py
import scipy.io
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from iso_forest import iso_forest
import iso_forest as IF

sys.setrecursionlimit(1000000)

metrics = ['AUC', 'TRAINING_TIME', 'TESTING_TIME', 'TREE_DEPTH',
                     'LEAF_COUNT', 'SPLIT_NODE_AVERAGE', 'TOTAL_TREE_AVERAGE']
# argparse for command-line arguments like config file location
parser = argparse.ArgumentParser(
    prog="Intel Optimized Data Discretion Isolation Forest Classifier (Intel ISOForest)",
    description="Create a isolation forest classifier model with data automatically optimzed for data discretion. \
        Optimzied Data Discretion bins your input data to create a smaller, faster, and highly accurate model tailored to your use case.",
    epilog= "Copyright 2023 Intel Corporation. Distributed under (license)")
parser.add_argument('config_filepath')

# Parse provided TOML config file
def parse_config_file(filepath):
    try:
        try:
            with open(filepath, 'rb') as f:
                config = tomllib.load(f)
                return config
        except tomllib.TOMLDecodeError:
            
            print("Error decoding TOML file - please check provided file.")
            exit()
    except OSError:
        print("Problem opening provided file: " + filepath)
        exit()

def load_data(config, filepath=""):
    choice = config["dataset"]
    if choice == "CLUSTER":
        n_samples_train, n_samples_test, n_outliers = 200, 70, 40
        rng = np.random.RandomState(0)
        covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
        cluster_1_train = 0.4 * \
            rng.randn(n_samples_train, 2) @ covariance + np.array([2, 2])
        cluster_1_test = 0.4 * \
            rng.randn(n_samples_test, 2) @ covariance + np.array([2, 2])
        cluster_2_train = 0.3 * \
            rng.randn(n_samples_train, 2) + np.array([-2, -2])
        cluster_2_test = 0.3 * \
            rng.randn(n_samples_test, 2) + np.array([-2, -2])
        outliers = rng.uniform(low=-8, high=-6, size=(n_outliers, 2))
        train = np.concatenate([cluster_1_train, cluster_2_train])
        test_nom = np.concatenate([cluster_1_test, cluster_2_test])
        test_ano = outliers
    if choice == "CUSTOM":
        # Load in csv
        test_nom, test_ano = np.array(), np.array()
        try:
            header = [""]
            try:
                header = str.split(config["dataset_header"], sep=",")
            except Exception as e:
                print("An issue with the provided header. Ensure your header is provided as a comma-seperated list, with no whitespace, like '1,2,3,class'. Error: ",e)
            f = pd.read_csv(config["filepath"], header=header)
        except Exception as e:
            print("Error with provided filepath + ", e)
            exit()
        # Split based on train, anomaly and nominal class
        train, test = train_test_split(f, .7)
        # Get noml test data
        try:
            if len(config["testnomlclass"] != 1):
                nomlclasses = str.split(config["testnomlclass"], sep=",")
                for i in range(nomlclasses):
                    test_nom = (test_nom | test[(test['class'] == nomlclasses[i])])
            else:
                nomlclasses = int(config["testnomlclass"])
                test_nom = test[(test['class'] == nomlclasses)]
        except Exception as e:
            print("Error with provided test nominal class. Ensure it is provided as an integer or a comma-seperated list of integers. Error: ", e)
        # Get anom test data
        try:
            if len(config["testanomclass"] != 1):
                anomclasses = str.split(config["testanomclass"], sep=",")
                for i in range(anomclasses):
                    test_ano = (test_ano | test[(test['class'] == anomclasses[i])])
            else:
                anomclasses = int(config["testanomclass"])
                test_ano = test[(test['class'] == anomclasses)]
        except Exception as e:
            print("Error with provided test nominal class. Ensure it is provided as an integer or a comma-seperated list of integers. Error: ", e)
    return train, test_nom, test_ano


def plot_train(train_data):
    plt.scatter(train_data[:, 0], train_data[:, 1], label='Training Data')
    plt.title("Training Data")
    plt.legend()
    plt.show()
    plt.clf()


def plot_test(test_nor, test_ano):
    plt.scatter(test_nor[:, 0], test_nor[:, 1], label='Test Normal Data')
    plt.scatter(test_ano[:, 0], test_ano[:, 1], label='Test Anomaly Data')
    plt.title("Testing Data")
    plt.legend()
    plt.show()

# Area under the curve of the receiver operating characteristic helper calculator
def AUCROC(score_nom, score_ano):
    num_nom, num_ano, AUCROC = len(score_nom), len(score_ano), 0
    score_nom_sorted = np.sort(score_nom)
    score_ano_sorted = np.sort(score_ano)
    for i in range(num_ano):
        exit = 0
        for j in range(num_nom):
            if score_nom_sorted[j] >= score_ano_sorted[i] and exit == 0:
                AUCROC += ((1/num_ano) * (j/num_nom))
                exit = 1
        if exit == 0:
            AUCROC += (1/num_ano)
    return AUCROC


def prep_dataset_and_build_forest(config, data_set, tree, growth_type, sss_val = 256):
    train_data, test_nom, test_ano = load_data(config)
    if not data_set == "CUSTOM":
        sss_val = 512 if data_set in ("FOREST_COVER", "FOREST_COVER_KURT") else 256
    else:
        sss_val = sss_val
    IF_model = iso_forest(type_itree=tree, training_set=train_data,
                        grow_full_tree=growth_type, sub_samp_size=sss_val)
    time_start = time.time()
    IF_model = IF_model.fit(train_data)
    training_time = time.time() - time_start
    IF_model = IF_model.set_anomaly_threshold(0.5)
    time_start = time.time()
    _, score_nom = IF_model.predict(test_nom)
    _, score_ano = IF_model.predict(test_ano)
    testing_time = time.time() - time_start
    AUC = AUCROC(score_nom, score_ano)
    tree_depth, leaf_count_ave, split_node_ave, total_tree_ave = IF_model.get_spatial_metrics()
    try:
        try:
            joblib.dump(IF_model, "/storage/models/" + config["saved_model_name"] + "_" + tree + ".model", 3,5)
        except Exception as e:
            joblib.dump(IF_model, config["saved_model_name"] + "_" + tree + ".model", 3,5)
            print("error", e)
    except FileNotFoundError as e:
        print("error", e)
    return training_time, testing_time, AUC, tree_depth, leaf_count_ave, split_node_ave, total_tree_ave


def generate_forest_and_metrics(config, data_set, tree, growth_type):
    print(f"Type of Isolation Forest: {tree}")
    print(f"Fully Grown: {growth_type}")

    training_time, testing_time, AUCROC, tree_depth, leaf_count_ave, split_node_ave, total_tree_ave = prep_dataset_and_build_forest(config,
        data_set, tree, growth_type)
    
    print(f"AUCROC: {AUCROC}")
    print(f"TRAIN TIME: {training_time}")
    print(f"TEST TIME: {testing_time}")
    print(f"AVE TREE DEPTH: {tree_depth}")
    print(f"AVE LEAF COUNT: {leaf_count_ave}")
    print(f"AVE SPLIT NODE COUNT: {split_node_ave}")
    print(f"AVE TREE SIZE: {total_tree_ave}")
    return [AUCROC, training_time, testing_time, tree_depth, leaf_count_ave, split_node_ave, total_tree_ave]

# run_basic_model: build the desired model and provide a comparison between quant and non- data models
# Provdies optioning for selecting opt and non opt models just for generation.
def run_basic_model(config):
    # dataframe to hold metrics
    df = pd.DataFrame()
    df['METRICS'] = metrics
    for fully_grown_bool in [True]:
        # keeping for potential dataset comparison runs
        for data_set in [config["dataset"]]:
            if config["compare"]:
                for tree in [IF.RANDOM, IF.QUANT]:
                    data = generate_forest_and_metrics(config, data_set, tree, fully_grown_bool)
                    df[f"{data_set}_{tree}"] = data
            else:
                if config["generate_optimized_model"]:
                    data = generate_forest_and_metrics(config, data_set, IF.QUANT, fully_grown_bool)
                    df[f"{data_set}_QUANTIZED"] = data
                elif config["generate_classic_model"]:
                    data = generate_forest_and_metrics(config, data_set, IF.RANDOM, fully_grown_bool)
                    df[f"{data_set}_RANDOM"] = data
                else:
                    print("no model type chosen")
                    exit()
                    
    try:
        print("saving results to disk")
        try:
            ## Saved to container storage volume, to be read and returned by the API server
            df.to_csv('/storage/results/showcase_results.csv')
        except Exception as e:
            df.to_csv('showcase_results.csv')
            print("error", e)
    except Exception as e:
        print("error", e)

def inference(config, IF_model=None):
    print("running inference")
    try:
        # load in model
        if IF_model == None:
            try:
                IF_model = joblib.load(config["path"])
            except FileNotFoundError:
                print("model not found, exiting...")
                exit(0)
        if IF_model == "empty":
            print(0)
            return
        
        # load in data
        print("running")
        train_data, test_nom, test_ano = load_data(config)
        print("test data loaded")
        _, nom_acc = IF_model.predict(test_nom)
        print("tested nominal data")
        _, ano_acc = IF_model.predict(test_ano)
        print("tested acc data")
        AUCROC_score = AUCROC(nom_acc, ano_acc)
        print(f"AUCROC : {AUCROC_score}")
        

    except Exception as e:
        print(e)

def main():
    args = parser.parse_args()
    config = parse_config_file(args.config_filepath)
    if config["task"] == "showcase":
        try:
            run_basic_model(config)
        except Exception as e: 
            print(e)
            exit()
    if config["task"] == "infer":
        try:
            inference(config)
        except Exception as e: 
            print(e)
            exit()

if __name__ == "__main__":
    main()
