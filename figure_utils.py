"""
Some random code that I have found to be useful for plotting figures.
This should become part of the main repo at some point, I will leave it out for now.

- David
"""

from natsort import natsorted

# Here is how I generate the general dictionary parameter for plots
COLORS = {
    "Coffee": "#97b4ff",
    "Salted Caramel": "#97b4ff",
    "Neapolitan": "#648fff",
    "Peanut Butter": "#1858ff",
    "Cherry": "#b3a4f7",
    "Pina Colada": "#8c75f2",
    "Chocolate": "#785ef0",
    "Cookie Dough": "#512fec",
    "Chocolate Chip": "#3d18e9",
    "Vanilla": "#e35299",
    "Mango": "#dc267f",
    "Black Raspberry": "#ff8032",
    "Rocky Road": "#fe6100",
    "Ground Truth": "#ffb000",
    "Mint Chocolate Chip": "#ffb000",
    "Bubble Gum": "#ffb000",
}

PLOT_SETUP = {
    "Salted Caramel": {"category": "1", "marker": "o"},
    "Neapolitan": {"category": "1", "marker": "v"},
    "Peanut Butter": {"category": "1", "marker": "^"},
    "Coffee": {"category": "1", "marker": "<"},
    "Cherry": {"category": "2", "marker": "o"},
    "Pina Colada": {"category": "2", "marker": "v"},
    "Cookie Dough": {"category": "2", "marker": "^"},
    "Chocolate Chip": {"category": "2", "marker": "<"},
    "Chocolate": {"category": "2", "marker": ">"},
    "Vanilla": {"category": "3", "marker": "o"},
    "Mango": {"category": "3", "marker": "v"},
    "Rocky Road": {"category": "4", "marker": "o"},
    "Black Raspberry": {"category": "4", "marker": "v"},
    "Ground Truth": {"category": "5", "marker": "o"},
    "Bubble Gum": {"category": "5", "marker": "v"},
    "Mint Chocolate Chip": {"category": "5", "marker": "^"},
}

for key in list(PLOT_SETUP.keys()):
    # PLOT_SETUP[key]["color"] = COLORS[PLOT_SETUP[key]["category"]]
    PLOT_SETUP[key]["color"] = COLORS[key]


# These two functions are useful when setting the order of how to plot figures
def compare_strings(fixed_string, other_string):
    return other_string.startswith(fixed_string)


def sort_labels_category(labels, plot_setup):
    labels_sorted = []
    for i in range(5):  # there are 5 categories
        for label in labels:
            if plot_setup[label]["category"] == str(i + 1):
                labels_sorted.append(label)

    return labels_sorted


labels = ...  # get labels from somwhere (pipeline results for example)

# This is the particular plot_setup for your data
plot_setup = {}
for i, label in enumerate(labels):
    for (
        possible_label
    ) in PLOT_SETUP.keys():  # generalized for labels like FLAVOR 1, FLAVOR 2, etc.
        # print(label, possible_label)
        if compare_strings(possible_label, label):
            plot_setup[label] = PLOT_SETUP[possible_label]

for label in labels:
    if label not in plot_setup.keys():
        raise ValueError(f"Label {label} not found in PLOT_SETUP")

labels = sort_labels_category(natsorted(labels), plot_setup)


# Then I do something like this, which let's me configure how the
# labels will be displayed in the plot
labels_for_plot = {
    "Neapolitan": "Neapolitan R1",
    "Neapolitan 2": "Neapolitan R2",
    "Peanut Butter": "Peanut Butter R1",
    "Peanut Butter 2": "Peanut Butter R2",
    "Salted Caramel": "Salted Caramel R1",
    "Salted Caramel 2": "Salted Caramel R2 1",
    "Salted Caramel 3": "Salted Caramel R2 2",
    "Chocolate": "Chocolate R1",
    "Chocolate 2": "Chocolate R2",
    "Chocolate Chip": "Chocolate Chip R1",
    "Cookie Dough": "Cookie Dough R1",
    "Cookie Dough 2": "Cookie Dough R2",
    "Pina Colada 1": "Piña Colada R2",
    "Mango": "Mango R1",
    "Vanilla": "Vanilla R1",
    "Vanilla 2": "Vanilla R2",
    "Black Raspberry": "Black Raspberry R1",
    "Black Raspberry 2": "Black Raspberry R2",
    "Rocky Road": "Rocky Road R1",
}
