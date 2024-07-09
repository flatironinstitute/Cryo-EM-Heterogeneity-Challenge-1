# TUTORIALS


## Brief overview

There are notebook tutorials for the each step in the analysis pipeline.

1. `1_tutorial_preprocessing.ipynb`
- input: folders with respective 80 `.mrc` files and a `populations.txt` file
- output: anonymized `submission_?.pt` files with aligned populations
This notebok walks through how to align the raw submissions, and the various options in this step.

2. `2_tutorial_svd.ipynb`
- input: `submission_?.pt` files
- output: `svd_results.pt`

This notebook walks through generating and analyzing (plots) the SVD results

3. `3_tutorial_map2map.ipynb`
- input: one `submission_?.pt` file
- output: a `.pkl` file

This notebook walks through generating and analyzing (plots) the map to map distance matrix results

4. `4_tutorial_distribution2distribution.ipynb`
- input: one `.pkl` file from the map2map step
- output: a `.pkl` file

4. `5_tutorial_plotting.ipynb`

This notebook walks through parsing and analyzing (plots) the map to map and distribution to distribution results.