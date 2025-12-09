import sys
dice_path = "/Users/volk/Documents/bau24-25/thesis/repos/DiCE-X"
sys.path.insert(0, dice_path)

from prefect import flow, task, get_run_logger
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime    # type: ignore
from sqlalchemy.ext.declarative import declarative_base    # type: ignore
from sqlalchemy.orm import sessionmaker    # type: ignore

from dice_ml_x.benchmarking import Benchmarking
from dice_ml_x.utils import helpers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@task
def load_datasets() -> dict | None:
    pass


@task
def generate_cfs_and_save_metrics() -> dict | None:
    pass


@task
def save_explainers_history_plot() -> None:
    pass


@task
def 


@flow(name="Dice-X-Experiment-Flow")
def dice_x_exp_flow() -> None:
    pass

if __name__ == "__main__":
    dice_x_exp_flow()