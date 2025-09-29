<h1 align='center'>Demo for the Cryo-EM Heterogeneity Challenge</h1>

<p align="center">


In this demo we show how the full analysis pipeline can be executed on a new submission. To do this, we have prepared a test system with reduced dimensionality. All the required files to run this demo can be accessed via the Open Science Foundation project [The Inaugural Flatiron Institute Cryo-EM Heterogeneity Community Challenge](https://osf.io/8h6fz/).


# Installation

First install our library by following the instructions on [our repository](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1). For completeness we provide simplified instructions here.

Create and activate a virtual environment using your prefered method. For example, with the built-in venv module:

```bash
python -m venv .venv
source .venv/bin/activate
```
Note: For this to work you need to have Python installed with version 3.10-3.13.

Then install our library by either cloning our repository and installing it locally, or by running:

```bash
pip install git+https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1.git
```

# Downloading the required data
The data can be download directly from the OSF link provided above, or by running the following command from the terminal. After downloading unzip the file and `cd` into the decompressed directory.

```bash
wget https://osf.io/download/68dacd9aabd55e9e99e4a4b6/ -O demo.zip
unzip demo.zip
cd demo
```

The demo data includes:
* `new_submission`: a directory with volumes and populations in the same format that was requested for participants of the challenge
* `ground_truth_volumes`: a subset of the ground truth (GT) volumes used to generate the simulated dataset used for the challenge.
* `1_preprocessing`: config files required to run the preprocessing of the new submission, as well as the GT mock submissions already preprocessed (Averaged GT and Sampled GT)
* `2_svd_analysis`: Config file to run the SVD analysis pipeline for the new submission and the GT mock submissions. A script to generate the plots based on the analysis results. Expected results are provided in `./2_svd_analysis/expected_results`.
* `3_map_to_map_analysis`: Config file to run the Map-to-Map pipeline for the new submission.
* `4_distribution_to_distribution_analysis`: config file to run the Distribution-to-Distribution analysis given the results of the Map-to-Map analysis.

For simplicity all volumes have been downsampled to a box size of 32 pixels. To reproduce the results in the paper please refer to our [Tutorial notebooks](https://github.com/flatironinstitute/Cryo-EM-Heterogeneity-Challenge-1/tree/main/tutorials).


# Running the pipelines

After downloading and unzipping the data, and installing our library; each pipeline can be run by executing the following commands:

Note: Running all the instructions will take several minutes. No GPU or special hardware is required.

## 1. Preprocessing

```bash
run_preprocessing --config 1_preprocessing/run_preproc_config.yaml
```

This should create the following files under `1_preprocessing/`:
1. `submission_icedemo_1.pt`: the preprocessed new submission, which was assigned ice cream flavor IceDemo. This file includes the preprocessed volumes, the submitted populations, the submission id (ice cream flavor), and the rotation matrix applied to align the volumes to the given reference structure `1_preprocessing/reference_volume.mrc`.

2. `submission_to_icrecream_table.json`: a json file that shows which submissions are assigned to each ice cream flavor. Useful when preprocessing multiple submissions.

## 2. SVD Analysis

To run the SVD analysis pipeline using the newly preprocessed submission, as well as the given GT mock submissions, please execute the following command in the terminal:
```bash
run_svd_pipeline --config 2_svd_analysis/config_svd.yaml
```
This will run the SVD analysis pipeline. The results can be found in `2_svd_analysis/svd_results`. To generate plots based on these results please run the following script
```bash
python 2_svd_analysis/plot_svd_results.py
```
The generated plots should be in `2_svd_analysis/svd_results` and should match the results in `2_svd_analysis/expected_results` up to a change in sign in the common embedding plots.

## 3. Map-to-Map analysis

Similarly, running the Map-to-Map analysis pipeline can be done by running the following command:

```bash
run_map_to_map_pipeline --config 3_map_to_map_analysis/config_map_to_map.yaml
```

## 4. Distribution-to-Distribution Analysis

Lastly, to run the Distribution-to-Distribution given the results from the previous step please run:

```bash
run_distribution_to_distribution_pipeline --config 4_distribution_to_distribution_analysis/config_distribution_to_distribution.yaml
```
