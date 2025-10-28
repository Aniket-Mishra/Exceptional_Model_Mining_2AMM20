Repo for EMM Project
Exceptional Model Mining on Model Residuals: Balancing Interpretability and Expressiveness in Rich Description Languages

This repo contains the code for the project.
We need to look at pipeline_v2 for the code. Pipeline was the trail.

To reproduce the results, we need to do the following, after having a python environment setup:

`python3 1_dataset_setup.py
python3 2_dataset_shape.py
python3 run_and_compare.py`

The commands will setup the data and then run the EMM and create your results. Once its done running, you can run:

`python3 3_create_plots.py`

It will generate the comparison plots. 