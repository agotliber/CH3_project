import os
import torch
import pickle
import numpy as np
from conf import *
import pandas as pd
from model import *
from clearml import Task
# from datetime import datetime
import matplotlib.pyplot as plt
from run import *


def get_test_data(path):
    test_labels = pd.read_csv(path + lables)
    test_exp = pd.read_csv(path + exp)
    test_dist = pd.read_csv(path + dist)
    test_seq = pd.read_csv(path + seq)
    return test_labels, test_exp, test_dist, test_seq


def create_clear_ml_task(project_name = "pytorch_ch3", task_type = "testing", task_name="test", model_name = "", test_path = ""):
    now = dt.datetime.now()
    dt_string = now.strftime("%d_%m_%H_%M_%S")
    task = Task.init(project_name=project_name, task_type=task_type, task_name =task_name+dt_string)
#     task.upload_artifact('test.py', artifact_object='test.py')
#     task.upload_artifact(Conf.PATH + model_name, artifact_object=Conf.PATH + model_name)
    task.add_tags("test data")

    task.add_tags(dt_string)
    task.set_comment(f"model path{Conf.PATH + model_name}")
    task.set_comment(f"test path{test_path}")
    print("Created clearml task")
    return task 

def run_test(labels, exp, dist, seq, test_path, train_label, model_name, fast_test=False, clear_ml_task= None, read_test_from_pickle = False):
    batch_size = 50
    if clear_ml_task == None:
         task = create_clear_ml_task()
    else: 
        task = clear_ml_task
    
    task.add_tags(train_label)
    
    if read_test_from_pickle:
        test = load_from_pickle(Conf.PATH + train_label + f"/test.pkl")
    else:         
        train, validation, test, validation_ch3_blind, test_ch3_blind, validation_e_blind, test_e_blind = Dataset.read_data_sets(
            filename_sequence= seq,
            filename_expression= exp,
            filename_labels= labels,
            filename_dist= dist,
            train_portion_subjects=0,
            train_portion_probes=0, validation_portion_subjects=0,
            validation_portion_probes=0, directory=test_path, is_prediction=False,
            load_model_ID=0)
        
    model = MultiModel()
    model_path = Conf.PATH + train_label + model_name
#     model.load_state_dict(torch.load(model_path + ".pt"))
    model = load_from_pickle(model_path + ".pkl")
    return run_test_eval(model, test, batch_size, task, fast_test)
    

if __name__ == "__main__":
    labels =  "test_labels.csv"
    exp =  "test_exp.csv"
    dist =  "test_dist.csv"
    seq =   "test_seq.csv"
    
    train_label = '08_17_18_59_45'

    model_name = "/ch3 full model training_final"
    test_path = Conf.PATH
    run_test(labels, exp, dist, seq, test_path, train_label, model_name,fast_test=True, read_test_from_pickle = False)
    