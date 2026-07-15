# Exceptional Model Residual Mining, and Three Richer EMM Description Languages

This repo contains the code the the paper "Exceptional Model Residual Mining, and Three Richer EMM Description Languages". The frozen code submitted to the IDA 2026 conference can be found in the branch: "IDA_2026_Submission".

DOI: https://doi.org/10.1007/978-3-032-23833-7_20

To replicate the results in the paper the following steps need to be followed:

1. Load a python environment
2. Run the code inside the pipeline folder by following the instructions below
3. The results will be saved locally.

### Replication Steps:

```
python3 1_dataset_setup.py

python3 2_dataset_shape.py

python3 run_and_compare.py
```

The commands will set up the data, run the EMM algorithm, and finally create the results. Once it's done running, you can run the code below to generate plots.

```
python3 3_create_plots.py
```
