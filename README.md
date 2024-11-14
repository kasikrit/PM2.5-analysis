# PM-analysis
Deep Learning and Statistical Approaches for Area-Based PM2.5 Forecasting in Hat Yai, Thailand

Abstact:

PM2.5 pollution poses a significant environmental and health concern across Southeast Asia, including Thailand. This study aims to forecast area-based PM2.5 concentrations in Hat Yai city, Songkhla province, using daily data collected from the Hat Yai monitoring station from 2013 to 2023. To achieve this, we developed and evaluated forecasting models utilizing both Deep Learning (DL) and Machine Learning (ML) techniques, with performance assessed using statistical tools. Among the tested models, the ConvLSTM1D-BiLSTM model, optimized with a 7-input window, 1-output window, and 6 strides, demonstrated the highest effectiveness, achieving a mean MAE of 2.05 and a mean R\(^2\) score of 0.68. In PM2.5 level classification, this model attained macro-average accuracy, sensitivity, and F1 scores of 0.86, 0.80, and 0.80, respectively. External validation using data from four nearby stations further confirmed the model's effectiveness, yielding a mean MAE of 1.38 and a mean R\(^2\) score of 0.90. These results underscore the robustness of our approach, supporting its practical application, particularly for local stations in Southern Thailand.

# Software requirements

The code and data for this study, developed using Python 3.8.18 and TensorFlow 2.13.1. Statistical analyses were conducted using Scipy (version 1.10.1), Statsmodels (version 0.14.1), and Scikit-posthocs(version 0.8.1). Additionally, the Kaggle Notebook illustrating the development and validation of the proposed model is publicly available at https://www.kaggle.com/code/kasikrit/pm-analysis-train-and-validate.
## Create a python environment
We use [Anaconda](https://anaconda.org/). With in Anaconda command prompt.

conda create -n py38pm python=3.8.18
conda activate py38pm

## Install software packages
Install software packages in requirements.txt
For example.

pip install TensorFlow==2.13.1 scipy=1.10.1, statsmodels==0.14.1 

## Dataset prepareration


# 1. How to train and validate a model?







