# PM-analysis
**Title:** Deep Learning and Statistical Approaches for Area-Based PM2.5 Forecasting in Hat Yai, Thailand

**Abstact:**

PM2.5 pollution poses a significant environmental and health concern across Southeast Asia, including Thailand. This study aims to forecast area-based PM2.5 concentrations in Hat Yai city, Songkhla province, using daily data collected from the Hat Yai monitoring station from 2013 to 2023. To achieve this, we developed and evaluated forecasting models utilizing both Deep Learning (DL) and Machine Learning (ML) techniques, with performance assessed using statistical tools. Among the tested models, the ConvLSTM1D-BiLSTM model, optimized with a 7-input window, 1-output window, and 6 strides, demonstrated the highest effectiveness, achieving a mean MAE of 2.05 and a mean R\(^2\) score of 0.68. In PM2.5 level classification, this model attained macro-average accuracy, sensitivity, and F1 scores of 0.86, 0.80, and 0.80, respectively. External validation using data from four nearby stations further confirmed the model's effectiveness, yielding a mean MAE of 1.38 and a mean R\(^2\) score of 0.90. These results underscore the robustness of our approach, supporting its practical application, particularly for local stations in Southern Thailand.

# Software Requirements

The code and data for this study were developed using [Python 3.8.18](https://www.python.org/downloads/release/python-3818/) and [TensorFlow 2.13.1](https://www.tensorflow.org/). Statistical analyses were conducted using [SciPy](https://scipy.org/) (version 1.10.1), [Statsmodels](https://www.statsmodels.org/stable/index.html) (version 0.14.1), and [Scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) (version 0.8.1). Additionally, the Kaggle Notebook illustrating the development and validation of the proposed model is publicly available at [this link](https://www.kaggle.com/code/kasikrit/pm-analysis-train-and-validate).


## Creating a Python Environment

We use [Anaconda](https://anaconda.org/). In the Anaconda command prompt, use the following commands:

```bash
conda create -n py38pm python=3.8.18
conda activate py38pm
```

## Installing Software Packages
Install software packages from [requirements.txt](https://github.com/kasikrit/PM-analysis/blob/main/requirements.txt). For example:

```bash
pip install TensorFlow==2.13.1 scipy==1.10.1 statsmodels==0.14.1
```

## Dataset Prepareration
Our method supports one-dimensional time series data in a CSV file (see [pm2013_2023.csv](https://github.com/kasikrit/PM-analysis/blob/main/pm2013_2023.csv)). You can prepare PM2.5 data with a column named PM. Example data format:

| Date       | Day | Month | Year | PM |
|------------|-----|-------|------|----|
| 2/1/2013   | 1   | Jan   | 2013 | 12 |
| 3/1/2013   | 2   | Jan   | 2013 | 15 |
| 4/1/2013   | 3   | Jan   | 2013 | 12 |
| 5/1/2013   | 4   | Jan   | 2013 | 17 |
| 6/1/2013   | 5   | Jan   | 2013 | 24 |
| 7/1/2013   | 6   | Jan   | 2013 | 25 |
| 8/1/2013   | 7   | Jan   | 2013 | 20 |
| 9/1/2013   | 8   | Jan   | 2013 | 16 |

You can add other columns, but we primarily use PM concentration values.

# Training the ConvLSTM1D-BiLSTM Model

Use the [pm-rescaled-DL-train.py](https://github.com/kasikrit/PM-analysis/blob/main/pm-rescaled-DL-train.py) file with the following hyperparameter settings. In this work, we use a 7-input window, 1-output window, and a 6-stride slicing window configuration for reading data points.

```python
# Define training parameters for the ConvLSTM1D-BiLSTM model
init_lr = 1e-4         # Initial learning rate for model training
epochs = 50            # Number of epochs (iterations over the entire dataset) for training
batch_size = 2         # Number of samples per batch (helps in optimizing model training)

# Model name for saving/loading purposes
model_name = 'Model-4'  # This is the best-performing model configuration, ConvLSTM1D-BiLSTM

# Convert the prepared PM2.5 time series data into a numpy array
time_series_data = np.array(pm_rescaled_1)  # Array containing the rescaled PM2.5 data for model input

# Window and stride configurations for input/output data sequences
W_ins = [7]             # Input window sizes (number of time steps considered for each input)
W_outs = [1]            # Output window sizes (number of future time steps the model will predict)
Strides = [1, 2, 3, 4, 5, 6, 7]  # Possible strides for the sliding window to adjust data overlap

```

# Validating the ConvLSTM1D-BiLSTM Model with External Unseen Data
1. **Prepare Data**: Format data points in CSV format (see [PM2.5(2023)-selected-app.csv](https://github.com/kasikrit/PM-analysis/blob/main/PM2.5(2023)-selected-app.csv)) with columns named after your station code or ID. Example:

| Date     | 44T | 62T | 63T | 78T | 80T |
|----------|-----|-----|-----|-----|-----|
| 5/2/23   | 11  | 17  | 20  | 14  | 12  |
| 6/2/23   | 11  | 16  | 19  | 11  | 15  |
| 7/2/23   | 14  | 16  | 20  | 10  | 18  |
| 8/2/23   | 13  | 21  | 23  | 14  | 16  |
| 9/2/23   | 13  | 22  | 19  | 13  | 14  |
| 10/2/23  | 13  | 19  | 20  | 10  | 16  |
| 11/2/23  | 10  | 15  | 22  | 14  | 15  |
| 12/2/23  | 12  | 17  | 18  | 14  | 15  |

2. **Load the Model**: Download the ConvLSTM1D-BiLSTM model at [Model-ConvLSTM1D-BiLSTM-pm-7-1-6-bs2-loss-mse-50ep-20230117.hdf5](https://github.com/kasikrit/PM-analysis/blob/main/Model-ConvLSTM1D-BiLSTM-pm-7-1-6-bs2-loss-mse-50ep-20230117.hdf5).
3. **Run Validation**: Use the [pm-rescaled-DL-val.py](https://github.com/kasikrit/PM-analysis/blob/main/pm-rescaled-DL-val.py) file with the following hyperparameter settings:
```python
# Load the pre-trained ConvLSTM1D-BiLSTM model with specific hyperparameters (7-1-6 configuration)
backup_model_best = 'Model-ConvLSTM1D-BiLSTM-pm-7-1-6-bs2-loss-mse-50ep-20230117.hdf5'
model = load_model(backup_model_best)  # Load the model from file
print(model.summary())  # Display the model architecture and parameters
print('Loaded:', backup_model_best)  # Confirm successful loading of the model

# File path to the PM2.5 data for validation, which includes PM2.5 data for selected stations in 2023
pm_file = 'PM2.5(2023)-selected-app.csv'

# List of station codes included in the dataset (columns for each station in the CSV file)
stations = ['80T', '63T', '78T', '62T']

# Define model configuration parameters
W_in = 7        # Input window size (number of previous time steps considered as input)
W_out = 1       # Output window size (number of future time steps predicted)
stride = 6      # Stride length for sliding window (determines how far the window shifts for each step)
N_forecast = 14 # Forecast horizon (number of time steps to predict into the future)

# Calculate the total length of input sequences based on model configuration
input_seqs = N_forecast * stride + W_in + W_out - 1  # Total number of time steps for input sequence
```


