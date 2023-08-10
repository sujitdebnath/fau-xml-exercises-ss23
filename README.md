> **Disclaimer:** The exercises and solutions for the xML course are property of [Simon Bachhuber](https://www.linkedin.com/in/simon-bachhuber-5b667216a/) and [Prof. Dr.-Ing. Thomas Seel](https://www.aibe.tf.fau.de/person/prof-dr-thomas-seel/) at [FAU Erlangen-Nürnberg](https://www.fau.eu/). Please be aware that copying content from here is your responsibility.

# FAU - Introduction to Explainable Machine Learning (xML) Exercises (SS23)

Welcome to the Introduction to Explainable Machine Learning (xML) exercises repository for the Summer'23 semester at [Friedrich-Alexander University Erlangen-Nürnberg](https://www.fau.eu/). This repository contains computer exercises of the xML course, taught by [Simon Bachhuber](https://www.linkedin.com/in/simon-bachhuber-5b667216a/), and [Prof. Dr.-Ing. Thomas Seel](https://www.aibe.tf.fau.de/person/prof-dr-thomas-seel/) at FAU Erlangen-Nürnberg.

## Setup Instructions

Follow these steps to set up your environment for the exercises:

1. Install [Python 3.x](https://www.python.org/).
2. Download and install [Anaconda](https://www.anaconda.com/download) Distribution on your machine.
3. Create a conda environment named `xML`.
```bash
conda create -n xML python=3.9
conda activate xML
```
4. Install mamba using conda.
```bash
conda install -c conda-forge mamba
```
5. Install other required packages from conda-forge.
```bash
mamba install -c conda-forge jupyterlab matplotlib pandas scikit-learn scipy statsmodels seaborn patsy numpy shap alibi "tokenizers>=0.11.1,!=0.11.3,<0.13"
```
6. Install other packages not available in conda-forge using pip. First, ensure pip is installed in the activated conda environment.
```bash
which pip
```
If it doesn't return `/anaconda3/envs/xML/bin/pip`, install pip.
```bash
conda install pip
```
Now install packages using pip.
```bash
pip install imodels tqdm PyALE lime
```
7. Downgrade numpy to version 1.23 due to compatibility with tensorflow 2.5.
```bash
pip install numpy==1.23
```
8. Open any exercise notebook, e.g., `ce1.ipynb`, `ce2.ipynb`, etc, and run the exercises.

Feel free to explore the world of Explainable Machine Learning through these exercises!
