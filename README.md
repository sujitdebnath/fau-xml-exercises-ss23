> **Disclaimer:** The computer exercises and solutions for xML course are property of [Simon Bachhuber](https://www.linkedin.com/in/simon-bachhuber-5b667216a/), and [Prof. Dr.-Ing. Thomas Seel](https://www.aibe.tf.fau.de/person/prof-dr-thomas-seel/) at [FAU Erlangen-Nürnberg](https://www.fau.eu/). Remember that copying content from here is your responsibility.

# FAU - Introduction to Explainable Machine Learning (xML) Exercises (SS23)

Welcome to the Introduction to Explainable Machine Learning (xML) exercises repository for the Summer'23 semester at [Friedrich-Alexander University Erlangen-Nürnberg](https://www.fau.eu/). This repository contains computer exercises of the xML course, taught by [Simon Bachhuber](https://www.linkedin.com/in/simon-bachhuber-5b667216a/), and [Prof. Dr.-Ing. Thomas Seel](https://www.aibe.tf.fau.de/person/prof-dr-thomas-seel/) at FAU Erlangen-Nürnberg.

## Setup for Exercises

1. Install [Python 3.x](https://www.python.org/)
2. Download and install [Anaconda](https://www.anaconda.com/download) Distribution into your machine
3. Create a conda environment, named xML
```bash
conda create -n xML python=3.9
conda activate xML
```
4. Install mamba using conda
```bash
conda install -c conda-forge mamba
```
5. Install other required packages from conda-forge
```bash
mamba install -c conda-forge jupyterlab matplotlib pandas scikit-learn scipy statsmodels seaborn patsy numpy shap alibi "tokenizers>=0.11.1,!=0.11.3,<0.13"
```
6. Install other packages which are not available in conda-forge; for those use pip. Before that please check is that pip installed in activated conda environment or not.
```bash
which pip
```
if it don't returns`/anaconda3/envs/xML/bin/pip`, then install pip first.
```bash
conda install pip
```
Now install packages using pip.
```bash
pip install imodels tqdm PyALE lime
```
7. Downgrade numpy, because tensorflow 2.5 requires numpy <= 1.23
```bash
pip install numpy==1.23
```
8. Open any exercise notebook, e.g., `ce1.ipynb`, `ce2.ipynb`, etc, and run.