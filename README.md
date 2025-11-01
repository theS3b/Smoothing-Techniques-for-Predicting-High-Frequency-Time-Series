<font size=4><div align='center'>[[📄 Tech Report](pdf/Smoothing-Techniques-for-Predicting-High-Frequency-Time-Series.pdf)]</div></font>

<h1 align="center">Smoothing Techniques for Predicting High-Frequency Time Series</h1>
<p align="center"><i>IIPP Lab (EPFL), ML Course Project, 2024–2025</i></p>

### 📝 Abstract

We explore methods to produce smoother high-frequency now-casts of a low-frequency indicator by leveraging available high-frequency data. Estimators often produce noisy predictions due to the scarcity of low-frequency observations and the volatility of high-frequency input data. To address this problem, we present various smoothing techniques applied before, after and during model training. Pre-processing approaches (e.g. data augmentation and Fourier-based upsampling) increase training targets. Post-processing approaches (e.g. ARIMA and Kalman filters) reduce noise in the model outputs. Finally, incorporating roughness penalties into the model learning objective further stabilises predictions. Together, these strategies significantly improve smoothing without compromising accuracy, thereby improving the practical reliability of now-casting.

## 🛠️ Set up

We provide a Conda environment with CUDA 12.1 support. On Windows (CMD):

```bat
conda env create -f environment.yml
conda activate ml-smoothing
```

Optional: verify CUDA availability inside the environment.

```bat
python -c "import torch; print(torch.cuda.is_available())"
```

The environment installs all required packages (including `filterpy`, `pmdarima`, `statsmodels`, `PyWavelets`, `torch`, etc.). See `environment.yml` for exact versions.

## 💽 Data

This repository includes ready-to-use data and outputs:

- `data/gdp/`: GDP CSVs
- `data/google_trends/`: Google Trends exports by country
- `data/output_for_paper/`: figures produced by notebooks (used in the paper)
- `paper_data/`: cached/processed data used to reproduce paper plots

To reproduce experiments with your own data, place files in the same folder structure and adjust paths in notebooks or the helpers in `utils/`.

## 💪 Methods and Experiments

This project is organized as self-contained notebooks. You can open each notebook and run all cells to reproduce the corresponding analysis.

### Post-processing denoising
- ARIMA smoothing: `Smoothing ARIMA.ipynb`, `Smoothing ARIMA auto.ipynb`
- Kalman filtering: `Kalman Filter on % GDP diff, with & without including real values.ipynb`
- Core implementations in `postprocessing/arima.py` and `postprocessing/KF.py`

### Pre-processing and augmentation
- Fourier-based upsampling and data augmentation: `Smoothing Fourrier Upsampling Data Augmentation.ipynb`
- Target interpolation augmentation: `Smoothing Y Interpolation Augmentation.ipynb`

### Objective-level smoothing
- Roughness penalties in the loss: `Smoothing Objective Function All Metrics.ipynb`, `Smoothing Objective Function Ord1,2 Only Beta 4.ipynb`
- Artifacts saved in `paper_data/` (e.g., `smoothing_objective_function_all_metrics_4_batchsize.pkl`)

### Modeling and analysis
- Neural networks and ensembling: `Simple NN Predictions Complete Model + Ensembling.ipynb`
- Gaussian processes: `GP Complete Mode.ipynb`
- Fourier analysis: `Fourier Smoothing Analysis.ipynb`
- Wavelet-based analysis: `Wavelet Analysis.ipynb`
- Performance visualization: `Plot Performance wr Params Objective Function.ipynb`, `Plot Performance wr Post Processing.ipynb`

## 📊 Metrics

We report both accuracy and smoothness:

- Accuracy: $R^2$ on the low-frequency target
- Smoothness: $e_i$ (incremental roughness) and $e_s$ (signal roughness)

Implementations live in `utils/results.py`. See the paper for precise definitions and discussion.

## 🗂️ Repository structure

- Root notebooks: each focuses on one technique or study and is runnable end-to-end
- `utils/`: data loading and preprocessing (`load_data.py`, `preprocessing_v2.py`, `GT_preprocessing.py`), modeling helpers (`neural_network.py`), and metrics (`results.py`)
- `postprocessing/`: ARIMA and Kalman filter implementations (`arima.py`, `KF.py`)
- `data/`: GDP and Google Trends inputs, plus figures for the paper under `output_for_paper/`
- `paper_data/`: intermediate artifacts used for figures and ablations
- `resources/`: project description and related references
- `ieee.mplstyle`: Matplotlib style used for figures

## 👥 Affiliation

This is a research project from the IIPP Lab at EPFL.

Supervision: Dr. Atin Aboutorabi, Dr. Yves Rychener

Contributors: Léonard Amsler, Sébastien Delsad, Nathan Gromb
 
