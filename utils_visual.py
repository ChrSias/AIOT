import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_instance_time_domain(df: pd.DataFrame):
    """Plots DataFrame columns over sample index in time domain."""
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20).legend(fontsize=18)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_instance_3d(df: pd.DataFrame, axes_list: tuple = ("acc_x", "acc_y", "acc_z")):
    """Plots three axes of a DataFrame in a 3D scatter."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs = df[axes_list[0]]
    ys = df[axes_list[1]]
    zs = df[axes_list[2]]
    ax.scatter(xs, ys, zs, color='green', s=50, alpha=0.6, edgecolors='w')
    ax.set_xlabel(axes_list[0])
    ax.set_ylabel(axes_list[1])
    ax.set_zlabel(axes_list[2])


def plot_np_instance(np_array: np.ndarray, columns_list: list):
    """Plots a NumPy array as a time-domain signal."""
    df = pd.DataFrame(np_array, columns=columns_list)
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_heatmap(df: pd.DataFrame):
    """Plots a heatmap of a DataFrame."""
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap='plasma')


def plot_scatter_pca(df: pd.DataFrame, c_name: str, cmap_set: str = "plasma"):
    """Scatter plot of PCA components coloured by label."""
    if len(df.columns) == 3:
        plt.style.use('classic')
        plt.figure(figsize=(16, 8))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df[c_name], cmap=cmap_set)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
    elif len(df.columns) == 4:
        plt.style.use('classic')
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[c_name], cmap=cmap_set)
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
    else:
        print("The DataFrame has more than 4 columns.")
