<h1 style='width: 100%; text-align: center;'>Machine Learning Project 2 - TPU Burners</h1>

<h3 style='width: 100%; text-align: center;'>
Smoothing Techniques for Predicting High-Frequency Time Series
</h3>

<br>

This repository contains the code for the second project of the Machine Learning (CS433) course at EPFL, in the fall semester of 2024.

We explore methods to produce smoother high-frequency now-casts of a low-frequency indicator by leveraging available high-frequency data. Estimators often produce noisy predictions due to the scarcity of low-frequency observations and the volatility of high-frequency input data. To address this problem, we present various smoothing techniques applied before, after and during model training. Pre-processing approaches (e.g. data augmentation and Fourier-based upsampling) increase training targets. Post-processing approaches (e.g. ARIMA and Kalman filters) reduce noise in the model outputs. Finally, incorporating roughness penalties into the model learning objective further stabilises predictions. Together, these strategies significantly improve smoothing without compromising accuracy, thereby improving the practical reliability of now-casting.

## Supervisors

- Atin Aboutorabi
- Yves Rychener

## Team members

- Léonard Amsler
- Sébastien Delsad
- Nathan Gromb

## Repository

The notebooks at the root of the repository contain our work on different smoothing techniques and are named accordingly. Each notebook can be run independently.

Our core functions are defined in the `utils/` folder, and in `postprocessing/` (for `KF.py`, `arima.py`).

The folder `utils/` provides the code used for data loading and preprocessing in addition to `results.py` which contains the code used to evaluate our models regarding $R^2$ score and the smoothness metrics implementations ($e_i$ and $e_s$, see the paper for more details).

In `data/`, you can find the dataset we used for the project (under `data/gdp/` and `data/google_trends/`) and the plots generated in the notebooks under `data/output_for_paper/` (sometimes based on data saved in `paper_data/`).

Finally, `resources/` contains the project proposal & description, as well as some of the papers that helped us in predicting the GDP on Google Trends.

## Python libraries

To run the notebooks, you will need the following libraries:

- `ipykernel`
- `numpy`
- `ipywidgets`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `tqdm`
- `PyWavelets`
- `pmdarima`
- `pandas`
- `joblib`
- `torch` (with cuda support)
- `filterpy` (from pip)

You can easily install them using the environment file provided in the repository (note that we're using cuda 12.1):

```bash
conda env create -f environment.yml
```
