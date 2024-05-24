# Analysis Notebooks

There are two notebooks in this directory, `master_paper_figures.ipynb` and `bonito_benchmarking.ipynb`. The former performs the analysis on decoding results to derive decoding accuracies for each studied algorithm, and the latter performs analysis for time benchmarking on a read by read basis to understand performance parameters for each soft decoder as well as memory overhead calculations. These two notebooks are sufficient to generate all figures, main and supplementary, in the submission. Included in this directory is a very simple conda environment file (`notebook-env.yml`) with a few basic dependencies needed to run the notebooks. The following commands will create this environment for you provided that you have some form of conda installed:

```
conda env create -f notebook-env.yml
```

Then, download the raw input data for the notebooks by running:

```
wget TODO:ADD LINK TO RELEASE DATA
tar -xzvf bioinformatics-notebook-data.tar.gz
```


Following this run the command:

```
conda activate dnastorage
jupyter-lab
```

Here you should be able to select a notebook and run its contents without much additional input to recreate the **Bioinformatics** submission figures.
