from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def combine_multiple_one_hot_multiclass(
    shap_values_multiclass: List[np.ndarray],
    feature_names: List[str],
    names: List[str],
    masks: List[np.ndarray],
) -> List[Tuple[np.ndarray, List[str]]]:
    """
    Combine SHAP values of multiple one-hot encoded features for multiclass classification.

    Args:
    shap_values_multiclass (List[np.ndarray]): List of numpy arrays of SHAP values, one for each class.
    feature_names (List[str]): List of feature names corresponding to shap_values.
    names (List[str]): List of names for the new combined features.
    masks (List[np.ndarray]): List of boolean arrays, each of the same length as the number of features.

    Returns:
    List[Tuple[np.ndarray, List[str]]]: A list of tuples, each containing a numpy
            array of combined SHAP values and a list of new feature names.
    """
    combined_shap_values_multiclass = []

    for shap_values in shap_values_multiclass:
        # Initialize new values array
        new_values_list = []

        # Non one-hot encoded features
        non_one_hot_mask = ~np.any(masks, axis=0)
        new_values_list.append(shap_values[:, non_one_hot_mask])

        # Process each one-hot encoded feature
        for _, mask in zip(names, masks):
            # Aggregate SHAP values for each one-hot encoded feature
            aggregated_values = shap_values[:, mask].sum(axis=1, keepdims=True)
            new_values_list.append(aggregated_values)

        # Concatenate the new values
        aggregated_values = np.concatenate(new_values_list, axis=1)

        # Update feature names
        new_feature_names = [
            feature_names[i] for i, included in enumerate(non_one_hot_mask) if included
        ] + names

        # Add to the list
        combined_shap_values_multiclass.append((aggregated_values, new_feature_names))

    return combined_shap_values_multiclass


def plot_confusion_matrix(
    cm: np.ndarray, classes: np.ndarray, ax: plt.Axes, title: str
) -> None:
    """
    Plots a confusion matrix with annotations.

    Args:
    cm (np.ndarray): Confusion matrix to be plotted.
    classes (np.ndarray): Array of class names.
    ax (plt.Axes): Matplotlib Axes object where the plot is drawn.
    title (str): Title of the plot.

    Returns:
    None
    """
    sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="Blues", cbar=True)

    # Set labels and title
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(title)
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    # Loop over data dimensions and create text annotations with contrast colors.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.5,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> None:
    """
    Evaluates a model and prints classification reports for training and
        testing datasets. Also, plots confusion matrices.

    Args:
    model: The model to be evaluated.
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training labels.
    X_test (np.ndarray): Testing features.
    y_test (np.ndarray): Testing labels.
    model_name (str): Name of the model for display purposes.

    Returns:
    None
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Classification report
    print(f"Classification Report for {model_name} (Train):")
    print(classification_report(y_train, y_train_pred, digits=4))
    print(f"Classification Report for {model_name} (Test):")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Confusion Matrices
    train_conf_mat = confusion_matrix(y_train, y_train_pred)
    test_conf_mat = confusion_matrix(y_test, y_test_pred)

    # Plotting Confusion Matrices
    _, ax = plt.subplots(1, 2, figsize=(15, 6))
    class_names = np.unique(y_train)  # Assuming class names are numerical and ordered
    plot_confusion_matrix(
        train_conf_mat, class_names, ax[0], f"Confusion Matrix - {model_name} (Train)"
    )
    plot_confusion_matrix(
        test_conf_mat, class_names, ax[1], f"Confusion Matrix - {model_name} (Test)"
    )

    plt.tight_layout()
    plt.show()
