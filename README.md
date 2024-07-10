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
The data is available via the Open Science Foundation project [The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge](https://osf.io/8h6fz/). You can download via a web browser, or programatically with wget as per [this script](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/main/tests/scripts/fetch_test_data.sh).

**_NOTE_**: We recommend downloadaing the data with the script and wget as the downloads from the web browser might be unstable.

# Installation

## Stable installation 
Installing this repository is simply. We recommend creating a virtual environment (using conda or pyenv), since we have dependencies such as PyTorch or Aspire, which are better dealt with in an isolated environment. After creating your environment, make sure to activate it and run

```bash
cd /path/to/Cryo-EM-Heterogeneity-Challenge-1
pip install .
```

## Developer installation
If you are interested in testing the programs previously installed, please, install the repository in development mode with the following commands:

```bash
cd /path/to/Cryo-EM-Heterogeneity-Challenge-1
pip install .[dev]
```

The test included in the repo can be executed with PyTest as shown below:

```bash
cd /path/to/Cryo-EM-Heterogeneity-Challenge-1
sh tests/scripts/fetch_test_data.sh # download test data from OSF
pytest tests/test_preprocessing.py
pytest tests/test_svd.py
pytest tests/test_map_to_map.py
pytest tests/test_distribution_to_distribution.py
```

# Running
If you want to run our code on the full challenge data, or you own local data, please complete the following steps

### 1. Download the full challenge data from [The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge](https://osf.io/8h6fz/)
You can do this through the web browser, or programatically with wget (you can get inspiration from [this script](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/main/tests/scripts/fetch_test_data.sh), which is just for the test data, not the full datasets)

### 2. Modify the config files and run the commands on the full challenge data
Point to the path where the data is locally
The [tutorial notebooks](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/tree/main/tutorials) explain how to setup the config files, and run the commands
```
cryo_challenge run_preprocessing                      --config config_files/config_preproc.yaml
cryo_challenge run_svd                                --config config_files/config_svd.yaml
cryo_challenge run_map2map_pipeline                   --config config_files/config_map_to_map.yaml
cryo_challenge run_distribution2distribution_pipeline --config config_files/config_distribution_to_distribution.yaml
```


# Contributing
If you find any bug or have a suggestion on the code feel free to open an issue [here](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/issues).

We also welcome any help with the development of this repository. If you want to contribute with your own suggestions, code, or fixes, we recommend creating a fork of this repository to avoid any incompatibilities with newer versions of the software. Once you are happy with your new code, please, make a PR from your fork to this repository.

We are also working on pipelines to simplify the exentension of the code with new metrics or functionalities, stay tuned!

# Acknowledgements
* Miro A. Astore, Geoffrey Woollard, David Silva-Sánchez, Wenda Zhao, Khanh Dao Duc, Nikolaus Grigorieff, Pilar Cossio, and Sonya M. Hanson. "The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge". 9 June 2023. DOI:10.17605/OSF.IO/8H6FZ
* [David Herreros](https://github.com/DavidHerreros) for testing and CI and debugging in this repo
