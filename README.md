<h1 align='center'>Cryo-EM Heterogeneity Challenge</h1>

<p align="center">

<img alt="Supported Python versions" src="https://img.shields.io/badge/Supported_Python_Versions-3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue">
<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/total">
<img alt="GitHub branch check runs" src="https://img.shields.io/github/check-runs/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/main">
<img alt="GitHub License" src="https://img.shields.io/github/license/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1">

</p>

<p align="center">

<img alt="Cryo-EM Heterogeneity Challenge" src="https://simonsfoundation.imgix.net/wp-content/uploads/2023/05/15134456/Screenshot-2023-05-15-at-1.39.07-PM.png?auto=format&q=90">

</p>



This repository contains the code used to analyse the submissions for the [Inaugural Flatiron Cryo-EM Heterogeneity Challenge](https://www.simonsfoundation.org/flatiron/center-for-computational-biology/structural-and-molecular-biophysics-collaboration/heterogeneity-in-cryo-electron-microscopy/).

# Scope
This repository explains how to preprocess a submission (80 maps and corresponding probability distribution), and analyze it. Challenge participants can benchmark their submissions locally against the ground truth and other submissions that are available on the cloud via the Open Science Foundation project [The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge](https://osf.io/8h6fz/).

# Warning
This is a work in progress, while the code will probably not change, we are still writting better tutorials, documentation, and other ideas for analyzing the data. We are also in the process of making it easier for other people to contribute with their own metrics and methods. We are also in the process of distributing the code to PyPi.

# Accesing the data
The data is available via the Open Science Foundation project [The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge](https://osf.io/8h6fz/). You can download via a web browser, or programatically with wget as per [this script](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/main/data/fetch_data.sh).

**_NOTE_**: We recommend downloadaing the data with the script and wget as the downloads from the web browser might be unstable.

# System Requirements

Running the code in this repository requires a system with Python 3.10-3.13. The code is tested automatically using GitHub actions for the latest stable Ubuntu version for Python 3.10, 3.11, 3.12 and 3.13. No specialized hardware is required for running the pipelines. GPU acceleration is possible through Torch's functionalities. The execution of the Map To Map pipelines can be accelerated through Dask, although this is optional.

# Installation

Note: the installation time may vary depending on the virtual environment framework used. Using python's default `venv` the installation should take between one and two minutes.

## Clone the repository
Before installation, please clone this GitHub repository locally:

```bash
git clone git@github.com:flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1.git
```

## Stable installation
The library in this repository may be installed with `pip`. We recommend creating a virtual environment (using conda or pyenv), since we have dependencies such as PyTorch or Aspire, which are better dealt with in an isolated environment. After creating your environment, make sure to activate it and run

```bash
cd /path/to/Cryo-EM-Heterogeneity-Challenge-1
pip install .
```

## Developer installation
If you are interested in developing, please, install the repository in editable mode and install the development dependencies with the following commands:

```bash
cd /path/to/Cryo-EM-Heterogeneity-Challenge-1
pip install -e ".[dev]"
```

# Testing

We recommend testing your installation by running

```bash
cd /path/to/Cryo-EM-Heterogeneity-Challenge-1
pytest tests/
```

This will make sure all the pipelines run on a mock set of submissions.

# Demo

We have prepared a demo dataset for running all our pipelines. The instructions for downloading the required data and running the pipelines can be found [here](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/tree/main/demo)

# Running
If you want to run our code on the full challenge data, or you own local data, please complete the following steps

### 0. Preprocessing raw submissions
As the the submissions for the challenge are anonymous, we do not provide the raw submissions. For running our preprocessing pipeline for new submissions please see our [preprocessing tutorial](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/main/tutorials/1_tutorial_preprocessing.ipynb). We provide access to all preprocessed submissions below.

### 1. Download the full challenge data from The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge
The data required to reproduce the results from the challenge can be downloaded from [OSF](https://osf.io/8h6fz/). Tje data consists of three folders: `dataset_1_submissions`, `datset_2_submissions`, and `Ground_truth`. The first two folders contain the preprocessed submissions for the experimental and simulated datasets, respectivelly. The third folder contains the GT-based mock submissions (Averaged GT and Sampled GT), as well as the volumes used for generating the data for the simulated datasets (`maps_gt_flat.pt`). These are provided as a .pt file to make easier the execution of our analysis pipelines.


### 2. Modify the config files and run the commands on the full challenge data
The [tutorial notebooks](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/tree/main/tutorials) explain how to setup the config files, and how to run each pipeline. We provide examples of config files for each pipeline. Given a config file, each pipeline can be executed from the command line as:

```bash
run_svd_pipeline                          --config path/to/config_files/config_svd.yaml
run_map_to_map_pipeline                   --config path/to/config_files/config_map_to_map.yaml
run_distribution_to_distribution_pipeline --config path/to/config_files/config_distribution_to_distribution.yaml
```

Example for config files can be found [here](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/tree/main/config_files)

### 3. SVD Pipeline

Running this pipeline for the simulated dataset submissions takes around 45 minutes on a computer node with 64 CPU cores and 128 GB of RAM. The runtime is divided into 1) Preparing the submissions (30 min) and 2) Computing the metrics (15 min). Since the prepared submissions are saved on disk (optional, see Tutorial), subsequent executions of this metric will be faster.

See our [SVD Tutorial](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/main/tutorials/2_tutorial_svd.ipynb) for a step-by-step on how to reproduce the results presented in the paper.


### 4. Map to Map Pipeline

Running this pipeline takes several hours or more per submission, depending on what metrics are requested. On a computer node with dozens of CPU cores and several hundreds of GB of RAM, the metrics `l2, bioem, corr` take several minutes; `fsc` takes several hours, with `res` taking minutes but needing the `fsc` results. The other metrics take much longer and it's recommended to only use a few hundred ground truth volumes for reference for them.

### 5. Distribution to Distribution Pipeline

Running this pipeline takes several minutes (we used a computer node with dozens of CPU cores and several hundreds of GB of RAM, but much less would be needed), and scales linearly with the number of replicates requested. Depending on the CVXPY solver requested, and the tolerance values requested, it can take a lot longer. We recommend ECOS, SCS, or CLARABEL.

# Contributing
If you find any bug or have a suggestion on the code feel free to open an issue [here](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/issues).

We also welcome any help with the development of this repository. If you want to contribute with your own suggestions, code, or fixes, we recommend creating a fork of this repository to avoid any incompatibilities with newer versions of the software. Once you are happy with your new code, please, make a PR from your fork to this repository.

We are also working on pipelines to simplify the exentension of the code with new metrics or functionalities, stay tuned!

# Acknowledgements
* Miro A. Astore, Geoffrey Woollard, David Silva-Sánchez, Wenda Zhao, Khanh Dao Duc, Nikolaus Grigorieff, Pilar Cossio, and Sonya M. Hanson. "The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge". 9 June 2023. DOI:10.17605/OSF.IO/8H6FZ
* [David Herreros](https://github.com/DavidHerreros) for testing and CI and debugging in this repo
