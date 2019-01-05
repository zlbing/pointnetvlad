import pickle
import os
import numpy as np
import random

filename = "generating_queries/oxford_evaluation_database.pickle"

with open(filename, 'rb') as handle:
	trajectories = pickle.load(handle)
	print("Trajectories Loaded.")
	print(trajectories)
