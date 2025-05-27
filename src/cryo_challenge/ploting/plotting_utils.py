import numpy as np


def res_at_fsc_threshold(fscs, threshold=0.5):
    res_fsc_half = np.argmin(fscs > threshold, axis=-1)
    fraction_nyquist = 0.5 * res_fsc_half / fscs.shape[-1]
    return res_fsc_half, fraction_nyquist


COLORS = {
    "Coffee": "#999999",
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
