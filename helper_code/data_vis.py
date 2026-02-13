import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from PIL import Image
import requests
from io import BytesIO
import math
import random
from matplotlib.lines import Line2D




#Plot the data in a two dimensional feature space
def plot_data(X, y, pcx=1, pcy=2,
              colors = {0 : "purple", 1 : "gold"} , 
              names = {0 : "Normal", 1 : "Abnormal"},
              size=(8, 6), highlight_idx=None, ax=None, title = None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    else:
        fig = ax.figure

    if title is not None:
        ax.set_title(title)

    custom_cmap = ListedColormap(list(colors.values()))
    
    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, pcx-1], X[:, pcy-1], s=40, c=y, label=list(map(lambda i: names[i], y)), cmap = custom_cmap, alpha=0.7)
    handles = [
                Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=colors[i], markersize=8, label=names[i])
                for i in colors
            ]

    #ax.legend(handles=handles, title="Classes", loc="upper right")

    if highlight_idx is not None:
        for idx in highlight_idx:
            ax.scatter(X[idx, pcx-1], X[idx, pcy-1],
                       s=80, facecolors="none", edgecolors="red", linewidths=2)
            
    return ax


def plot_density_by_class(X, y, pc=1,
                          colors={0: "purple", 1: "gold"},
                          names={0: "Normal", 1: "Abnormal"},
                          size=(8, 6), show_percentile=True,
                          legend_order=[1,0]):
                                 
    plt.figure(figsize=size)


    for cls in legend_order:
        pc_vals = X[y == cls, pc-1]

        plt.hist(
            pc_vals,
            bins=50,
            density=True,
            alpha=0.7,
            label=names[cls], 
            color=colors[cls]
        )

        if show_percentile:
            p = 20 if cls == 0 else 80
            threshold = np.percentile(pc_vals, p)

            plt.axvline(
                threshold,
                color=colors[cls],
                linestyle="--",
                linewidth=2,
                alpha=0.9
            )

            # annotate lines
            plt.text(
                threshold + 0.5,
                0.22,
                f"P{p}:({threshold:.2f})",
                color=colors[cls],
            )


    # plt.legend(loc='upper right', title="Classes")


def format_plot(title, xlabel, ylabel, png_name=None, save=True):
    plt.title(title, pad=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if save:
        plt.savefig(f"{png_name.replace(' ', '_').lower()}.png", dpi=300)
    plt.tight_layout()



#Plot the data in a two dimensional feature space and fit an SVM classifier to it
def plot_with_decision_boundary(
    kernel, X, y, ax=None, colors = {0 : "purple", 1 : "gold"}, names = {0 : "Normal", 1 : "Abnormal"}, title = None
):
    """
    kernel: Which kernel to use for the SVM classifier (rbf, linear, etc.)
    X: Embeddings
    y: Labels
    """
    svm_classifier = SVC(kernel=kernel, C=1, gamma='scale')
    svm_classifier.fit(X, y)

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    custom_cmap = ListedColormap(list(colors.values()))

    # Plot decision boundary and margins
    common_params = {"estimator": svm_classifier, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        cmap = custom_cmap
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    ax.scatter(
        svm_classifier.support_vectors_[:, 0],
        svm_classifier.support_vectors_[:, 1],
        s=150,
        facecolors="none",
        edgecolors="k",
    )


    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], s=30, c=y, label=list(map(lambda i: names[i], y)), cmap = custom_cmap, edgecolors="k")
    ax.legend(handles=scatter.legend_elements()[0], labels=["Normal", "Abnormal"], loc="upper right", title="Classes")

    ax.set_xlabel("Principal Axis 1")
    ax.set_ylabel("Principal Axis 2")

    if title is None:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(title)

    if ax is None:
        plt.show()

    #Show accuracy of predictions
    pred = svm_classifier.predict(X)

    print(f"Accuracy: {(pred == y).mean()}")

    return svm_classifier

def save_embeddings_to_json(
    reduced_embeddings, labels, image_urls, a_ids,
    output_path="embedding_data/d3_data.json"
):
    
    reduced_embeddings = np.array(reduced_embeddings)
    labels = np.array(labels)

    if reduced_embeddings.shape[1] != 2:
        raise ValueError("expected reduced_embeddings to have shape (n_samples, 2)")

    if not (len(reduced_embeddings) == len(labels) == len(image_urls)):
        raise ValueError("lengths of embeddings, labels, and image_urls must match")

    d3_data = [
        {
            "x": float(x),
            "y": float(y),
            "label": int(label),
            "img": str(img),
            "a_id": str(a_id)
        }
        for (x, y), label, img, a_id in zip(reduced_embeddings, labels, image_urls, a_ids)
    ]

    with open(output_path, "w") as f:
        json.dump(d3_data, f)

    print(f"saved embeddings to {output_path}!")


def display_images_grid(img_urls, cols=4, figsize=(20, 20)):

    n_images = len(img_urls)
    rows = math.ceil(n_images / cols)
    
    fig_width = cols * 8
    fig_height = rows * 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height),
                             gridspec_kw={'wspace':0.05, 'hspace':0.05})
    
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for ax, url in zip(axes, img_urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            ax.imshow(img, aspect='auto')  # fills the subplot
            ax.axis('off')
        except:
            ax.axis('off')
    
    # hide any extra axes
    for ax in axes[n_images:]:
        ax.axis('off')
    
    plt.show()