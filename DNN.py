
# INTRO TO DEEP LEARNING 
# Understanding Dense Layers and Activation Functions
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#  SETUP

proj_dir = os.path.dirname(os.path.abspath(__file__))

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=18,
    titlepad=10
)

print("TensorFlow version:", tf.__version__)
print("Setup complete!")
# LOAD THE CONCRETE DATASET


dataset_path = os.path.join(proj_dir, "data/concrete.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Cannot find dataset: {dataset_path}\n"
        "Place concrete.csv in the same folder as DNN.py."
    )

concrete = pd.read_csv(dataset_path)

print("\nDataset loaded successfully.")
print("Dataset shape:", concrete.shape)
print("\nFirst five rows:")
print(concrete.head())


#  INPUT SHAPE


# The target is the compressive strength of concrete.
target_column = "CompressiveStrength"

if target_column not in concrete.columns:
    raise ValueError(
        f"Target column '{target_column}' is missing."
    )

# Separate input features (X) and target values (y).
X = concrete.drop(columns=[target_column])
y = concrete[target_column]

# Input shape is the number of features in one row.
input_shape = [X.shape[1]]

print("\nEXERCISE 1: INPUT SHAPE")
print("Input shape:", input_shape)
print("Number of features:", X.shape[1])
print("Target:", target_column)


#  MODEL WITH HIDDEN LAYERS

# Each Dense layer calculates:
# output = activation(inputs @ weights + biases)
#
# ReLU activation:
# relu(x) = max(0, x)
#
# Three hidden layers, each containing 512 neurons.
# One output neuron predicts compressive strength.

model = keras.Sequential([
    layers.Input(shape=input_shape),

    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),

    # Linear output layer for regression.
    layers.Dense(1)
])

print("\nEXERCISE 2: MODEL WITH HIDDEN LAYERS")
model.summary()


# EXPLICIT ACTIVATION LAYERS

# Instead of putting activation="relu" inside Dense,
# define the activation as a separate layer.

model_explicit = keras.Sequential([
    layers.Input(shape=input_shape),

    layers.Dense(32),
    layers.Activation("relu"),

    layers.Dense(32),
    layers.Activation("relu"),

    layers.Dense(1)
])

print("\nEXERCISE 3: EXPLICIT ACTIVATION LAYERS")
model_explicit.summary()


# 4. COMPARE THE TWO ACTIVATION SYNTAXES

# These two approaches are equivalent in principle:
#
# Dense(32, activation="relu")
#
# and
#
# Dense(32)
# Activation("relu")

print("\nBoth models have been created successfully.")
print("No training has been performed yet.")
