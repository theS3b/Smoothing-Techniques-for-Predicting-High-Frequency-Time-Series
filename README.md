<h1 style='width: 100%; text-align: center;'>Machine Learning Project 2 - TPU Burners</h1>

<h3 style='width: 100%; text-align: center;'>
Smoothing Techniques for Predicting High-Frequency Time Series
</h3>

<br>

This repository contains the code for the second project of the Machine Learning (CS433) course at EPFL, in the fall semester of 2024.

The goal of the project is to explore methods to smooth high-frequency predicitions of a low-frequency time series. To apply the smoothing techniques, we use quarterly GDP data predicted on a monthly Google Trends.

## Supervisors

- Atin Aboutorabi
- Yves Rychener

## Team members

- Léonard Amsler
- Sébastien Delsad
- Nathan Gromb

## Repository

The notebooks at the root of the repository contain our work on different smoothing techniques and are named after the ones they explore. Each notebook can be run independently.

Our models are defined in the `utils/` folder, containing the reference neural network (`neural_network.py`), and in `postprocessing/` (`KF.py`, `arima.py`).

The folder `utils/` provides the code used for data loading and preprocessing in addition to `results.py` which contains the code used to evaluate our models regarding $R^2$ score and our own smoothness metric, a composition of different measures of time series smoothness.

In `data/`, you can find the dataset we used for the project (under `data/gdp/` and `data/google_trends/`) and the plots generated in the notebooks under `data/output_for_paper/` (sometimes based on data saved in `paper_data/`).

Finally, `resources/` contains the project proposal & description, as well as some of the papers that helped us in predicting the GDP on Google Trends.

## Python libraries

All the Python that we used can be found in the `requirements.txt` file.
