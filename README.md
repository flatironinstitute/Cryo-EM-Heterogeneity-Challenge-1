<h1 align='center'>Cryo-EM Heterogeniety Challenge</h1>

This repository contains the code used to analyse the submissions for the first Cryo-EM Heteorgeneity Challenge.


## Warning


This is a work in progress, while the code will probably not change, we are still writting better tutorials, documentation, and other ideas for analyzing the data. We are also in the process of making it easier for other people to contribute with their own metrics and methods. We are also in the process of distributiing the code to PyPi


## Accesing the data

The data is available via the Open Science Foundation project [The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge](https://osf.io/8h6fz/). You can download via a webbroswer, or programatically with wget as per [this script](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/blob/main/tests/scripts/fetch_test_data.sh).


## Installation

Installing this repository is simply. We recommend creating a virtual environment (using conda or pyenv), since we have dependencies such as PyTorch or Aspire, which are better dealt with in an isolated environment. After creating your environment, make sure to activate it and run

```bash
cd /path/to/Cryo-EM-Heterogeneity-Challenge-1
pip install .
```

## Running
If you want to run our code, please check the notebooks in the [tutorials folder](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/tree/main/tutorials).

The tutorials explain how to setup the config files, and run the commands
```
cryo_challenge run_preprocessing                      --config config_files/config_preproc.yaml
cryo_challenge run_svd                                --config config_files/config_svd.yaml
cryo_challenge run_map2map_pipeline                   --config config_files/config_map_to_map.yaml
cryo_challenge run_distribution2distribution_pipeline --config config_files/config_distribution_to_distribution.yaml
```

## Acknowledgements
* Miro A. Astore, Geoffrey Woollard, David Silva-Sánchez, Wenda Zhao, Khanh Dao Duc, Nikolaus Grigorieff, Pilar Cossio, and Sonya M. Hanson. "The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge". 9 June 2023. DOI:10.17605/OSF.IO/8H6FZ
* David Herreros for testing and CI and debugging in this repo
