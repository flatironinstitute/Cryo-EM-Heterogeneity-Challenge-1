'''
TODO: Need to implement this properly

def normalize_mean_std(vols_flat):
    """
    vols_flat.shape is (n_vols, n_pix**3)
    vols_flat is a torch tensor
    """
    return (vols_flat - vols_flat.mean(-1, keepdims=True)) / vols_flat.std(
        -1, keepdims=True
    )


def normalize_median_std(vols_flat):
    """
    vols_flat.shape is (n_vols, n_pix**3)
    vols_flat is a torch tensor
    """
    return (vols_flat - vols_flat.median(-1, keepdims=True).values) / vols_flat.std(
        -1, keepdims=True
    )
'''
