import numpy as np
from typing import Tuple, Optional


DEFAULT_PLOT_PARAMETERS = {
    "Salted Caramel": {"marker": "o", "color": "#648fff"},
    "Neapolitan": {"marker": "v", "color": "#648fff"},
    "Peanut Butter": {"marker": "^", "color": "#648fff"},
    "Coffee": {"marker": "<", "color": "#648fff"},
    "Cherry": {"marker": "o", "color": "#785ef0"},
    "Pina Colada": {"marker": "v", "color": "#785ef0"},
    "Cookie Dough": {"marker": "^", "color": "#785ef0"},
    "Chocolate Chip": {"marker": "<", "color": "#785ef0"},
    "Chocolate": {"marker": ">", "color": "#785ef0"},
    "Vanilla": {"marker": "o", "color": "#dc267f"},
    "Mango": {"marker": "v", "color": "#dc267f"},
    "Rocky Road": {"marker": "o", "color": "#fe6100"},
    "Black Raspberry": {"marker": "v", "color": "#fe6100"},
    "Ground Truth": {"marker": "o", "color": "#ffb000"},
    "Sampled GT": {"marker": "v", "color": "#ffb000"},
    "Averaged GT": {"marker": "^", "color": "#ffb000"},
}

DEFAULT_LABEL_ORDERING = [
    "Pina Colada 1",
    "Cookie Dough 1",
    "Cookie Dough 2",
    "Cherry 1",
    "Cherry 2",
    "Chocolate Chip 1",
    "Chocolate Chip 2",
    "Chocolate 1",
    "Chocolate 2",
    "Rocky Road 1",
    "Rocky Road 2",
    "Rocky Road 3",
    "Black Raspberry 1",
    "Black Raspberry 2",
    "Vanilla 1",
    "Vanilla 2",
    "Mango 1",
    "Salted Caramel 1",
    "Salted Caramel 2",
    "Salted Caramel 3",
    "Peanut Butter 1",
    "Peanut Butter 2",
    "Neapolitan 1",
    "Neapolitan 2",
    "Ground Truth",
    "Sampled GT",
    "Averaged GT",
]

FORMATTED_LABELS_FOR_PLOTS = {
    "Neapolitan 1": "Neapolitan 1",
    "Neapolitan 2": "Neapolitan 2",
    "Peanut Butter 1": "Peanut Butter 1",
    "Peanut Butter 2": "Peanut Butter 2",
    "Salted Caramel 1": "Salted Caramel 1",
    "Salted Caramel 2": "Salted Caramel 2",
    "Salted Caramel 3": "Salted Caramel 3",
    "Cherry 1": "Cherry 1",
    "Cherry 2": "Cherry 2",
    "Chocolate 1": "Chocolate 1",
    "Chocolate 2": "Chocolate 2",
    "Chocolate Chip 1": "Chocolate Chip 1",
    "Chocolate Chip 2": "(*) Chocolate Chip 2",
    "Cookie Dough 1": "Cookie Dough 1",
    "Cookie Dough 2": "Cookie Dough 2",
    "Pina Colada 1": "Piña Colada 2",
    "Mango 1": "Mango 1",
    "Vanilla 1": "Vanilla 1",
    "Vanilla 2": "Vanilla 2",
    "Black Raspberry 1": "Black Raspberry 1",
    "Black Raspberry 2": "Black Raspberry 2",
    "Rocky Road 1": "Rocky Road 1",
    "Rocky Road 2": "(*) Rocky Road 2",
    "Rocky Road 3": "(*) Rocky Road 3",
    "Sampled GT": "Sampled GT",
    "Averaged GT": "Averaged GT",
    "Ground Truth": "Ground Truth",
}

ABBREVIATIONS_FOR_LABELS = {
    "Pina Colada 1": "PC 2",
    "Cookie Dough 1": "CD 1",
    "Cookie Dough 2": "CD 2",
    "Cherry 1": "Ch 1",
    "Cherry 2": "Ch 2",
    "Chocolate Chip 2": "CC 2",
    "Chocolate Chip 1": "CC 1",
    "Chocolate 2": "C 2",
    "Chocolate 1": "C 1",
    "Rocky Road 3": "RR 3",
    "Rocky Road 2": "RR 2",
    "Rocky Road 1": "RR 1",
    "Black Raspberry 2": "BR 2",
    "Black Raspberry 1": "BR 1",
    "Vanilla 2": "V 2",
    "Vanilla 1": "V 1",
    "Mango 1": "M 1",
    "Salted Caramel 3": "SC 3",
    "Salted Caramel 2": "SC 2",
    "Salted Caramel 1": "SC 1",
    "Peanut Butter 2": "PB 2",
    "Peanut Butter 1": "PB 1",
    "Neapolitan 2": "N 2",
    "Neapolitan 1": "N 1",
    "Averaged GT": "Avg. GT",
    "Sampled GT": "Samp. GT",
    "Ground Truth": "GT",
}


def sort_labels_manually(
    labels: list[str] | np.ndarray, sorting_labels: Optional[list[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort labels given a predefined order.

    ** Arguments: **
        - labels: list of labels to sort
        - sorting_labels: predefined order of labels. If None, uses LABELS_ORDERED
    ** Returns: **
        - sorted_labels: list of sorted labels
        - indices: indices that sort the labels
    """

    labels = np.asarray(labels)

    if sorting_labels is None:
        sorting_labels = DEFAULT_LABEL_ORDERING

    try:
        sort_dict = {label: i for i, label in enumerate(sorting_labels)}
        indices = np.array([sort_dict[label] for label in labels])
    except KeyError as e:
        raise KeyError(f"Label {e} not found in sorting_labels") from e

    indices = np.argsort(indices)
    return labels[indices], indices


def _compare_strings(fixed_string, other_string):
    return other_string.startswith(fixed_string)


def get_plot_parameters_for_labels(
    labels: list[str] | np.ndarray, plot_parameters_dict: Optional[dict] = None
) -> dict[str, dict]:
    """
    Get plot parameters for a list of labels.

    ** Arguments: **
        - labels: list of labels to get plot parameters for
        - plot_parameters_dict: dictionary of plot parameters. If None, uses DEFAULT_PLOT_PARAMETERS
    ** Returns: **
        - plot_parameters: dictionary of plot parameters for each label
    """
    if plot_parameters_dict is None:
        plot_parameters_dict = DEFAULT_PLOT_PARAMETERS

    plot_parameters = {}
    for label in labels:
        for possible_label in plot_parameters_dict.keys():
            if _compare_strings(possible_label, label):
                plot_parameters[label] = plot_parameters_dict[possible_label]
    for label in labels:
        if label not in plot_parameters.keys():
            raise ValueError(f"Label {label} not found in plot_parameters_dict")
    return plot_parameters
