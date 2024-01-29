#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:39:29 2024

@author: kasikritdamkliang
"""
import tensorflow as tf
print(tf.__version__) 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from math import log

from livelossplot import PlotLossesKeras
import tensorflow_addons as tfa
tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
from tensorflow.keras import layers
from tensorflow.keras import backend, optimizers
# import mymodels
import keras

#%https://www.kaggle.com/code/rakshitacharya/convlstm1d-for-weather-data
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

def categorize(pm_list):
    pm_cat = []
    for pm in pm_list:
        if pm >= 0 and pm <= 15.0:
            level = 0
        elif pm > 15.0 and pm <= 25.0:
            level = 1
        elif pm > 15.0 and pm <= 25.0:
              level = 2
        elif pm > 25.0 and pm <= 37.5:
            level = 3
        elif pm > 37.5 and pm <= 75.0:
            level = 4
        else:
            level = 5
        pm_cat.append(level)
    return pm_cat
        

def create_dataset_with_stride(time_series, W_in, W_out, S):
    X, y = [], []
    for i in range(0, len(time_series) - W_in - W_out + 1, S):  # Notice the stride S in the range
        X.append(time_series[i : i + W_in])
        y.append(time_series[i + W_in : i + W_in + W_out])
    return np.array(X), np.array(y)

#%%%
pm_file = 'PM2.5(2023)-selected-app.csv'
df = pd.read_csv(pm_file)

# df = pd.read_csv('test-55-day.csv')
n = len(df)
print(n)
stations = ['80T', '63T', '78T', '62T']
# station = stations[1]
df_output = pd.DataFrame([])
#%
init_lr = 10e-4
epochs = 50
batch_size = 2
model_name = 'Model-4'

#7-1-6 model
backup_model_best ='../Model-ConvLSTM1D-BiLSTM-pm-7-1-6-bs2-loss-mse-50ep-20230117.hdf5'
model = load_model(backup_model_best)
print(model.summary())
print('Loaded: ', backup_model_best)

model.compile(
    optimizer=optimizers.Adam(
        learning_rate = init_lr),
    loss='mse',
    metrics=['mae', 'mse'])

#%% 
# W_ins = [7, 7, 7, 8, 15]  # The input window size from your table
# W_outs = [1, 3, 7, 2, 5]  # The output range from your table
W_in = 7
W_out = 1
stride = 6 
N_forecast = 14
input_seqs = N_forecast * stride + W_in + W_out - 1

# N_forecasts = [7, 14, 21, 28, 35]
# input_seqs = []
# for N_forecast in N_forecasts:
#     input_seq = N_forecast * stride + W_in + W_out - 1
#     print(input_seq)
#     input_seqs.append(input_seq)
    
# for pair in zip(N_forecasts, input_seqs):
#     print(pair)
  
for station in stations:
    pm = df[station].values #Satun
    plt.figure(dpi=300)
    plt.hist(pm, bins='auto')
    plt.title('PM2.5')
    #%\ pm=>rescaled
    sc = MinMaxScaler()
    pm_rescaled_1 = sc.fit_transform(pm.reshape(-1,1))
    time_series_data = np.array(pm_rescaled_1[:input_seqs]) 
      
    #%%
    # for cnt in range(5):
    #     print(cnt)   
    # for W_in, W_out in zip(W_ins, W_outs):
    data_note = f'pm-{W_in}-{W_out}-{stride}-bs{batch_size}-loss-mse'  
    num_classes = W_out
    print(data_note)
    
    X, y = create_dataset_with_stride(time_series_data, W_in, W_out, stride)
    y = y.reshape(-1, 1)
    print(X.shape, y.shape)
    
    #%%  
    X_test = np.reshape(X,
                    (X.shape[0],
                    1,
                    X.shape[1],
                    1)
                    )
    print(X.shape)
    
    y_test = np.array(pm_rescaled_1[input_seqs: input_seqs+14]) 
    print(y_test.shape)
    
    if W_out==1:
        y_test = y.reshape(-1, 1)
    print(y_test.shape)
    
    #%%   
    # inp = layers.Input(shape=(1, W_in, 1))
    # print(inp.shape)   
    
    # if model_name == 'Model-1':
    #     model = mymodels.build_model_1(inp, num_classes)  
    # elif model_name == 'Model-5':
    #     model = mymodels.build_model_5(inp, num_classes)    
    # else:
    #     model = mymodels.build_model_4(inp, num_classes)  
    
    # model.compile(
    #     optimizer=optimizers.Adam(
    #         learning_rate = init_lr),
    #     # optimizer='adam',
    #     loss='mse',
    #     # loss='mae',
    #     metrics=['mae', 'mse'])
    
    # print(model.summary())
      
    # backup_model_best = f'{model.name}-{data_note}-50ep-20230118.hdf5'
    # print('\nbackup_model_best: ', backup_model_best)
    # mcp2 = ModelCheckpoint(
    #     backup_model_best,
    #     save_best_only=True)


    #%%    
    y_pred = model.predict(X_test, verbose=1)
    
    #%%
    print("Predict PM values")
    y_test_reversed = sc.inverse_transform(
        y_test.reshape(-1,1))
    y_pred_reversed = sc.inverse_transform(
        y_pred.reshape(-1,1))
    
    residuals = abs(y_test_reversed-y_pred_reversed)
    # plt.hist(residuals)
    
    r2 = r2_score(y_test_reversed, y_pred_reversed)
    mse = mean_squared_error(y_test_reversed, y_pred_reversed)
    mae = mean_absolute_error(y_test_reversed, y_pred_reversed)
    rmse = np.sqrt(mse)
    
    print("MSE: ", mse)
    print("MAE: " , mae)
    print("RMSE: " , rmse)
    print("R2_score: ", r2) 
    
    #%%
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(12, 6), dpi=600)
    plt.plot(y_test_reversed.flatten(), color='blue')
    plt.plot(y_pred_reversed.flatten(), color='orange')
    plt.legend(["actual", "pred"])
    plt.title("Prediction of PM " + f"{model.name}-{station}")
    plt.xlabel(f"MAE: {mae:.2f}")
    # plt.savefig(f"{model.name}-{station}.png")
    plt.show()
       
     #%%
    df_output = df_output._append(pd.DataFrame(
        {
        "W_in": [W_in],
        "W_out": [W_out],
        "stride": [stride],
        "station": [station],
        "MSE": [mse],
        "MAE": [mae],
        "RMSE": [rmse],
        "R2_score": [r2]
        },
        index=[0]),
        ignore_index=True)
    print(df_output)

displacement = [62, 103, 152, 162]
df_output['displacement'] = displacement

df_output_file = f"{model.name}.csv"
df_output.to_csv(df_output_file)


#%%
# Number of stations and metrics
n_stations = len(df_output['station'])
n_metrics = 4  # MSE, MAE, RMSE, R2_score
# Create a figure and axis
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
fontsize=18
plt.rcParams.update({'font.size': fontsize})
# Set the positions for the bars
bar_width = 0.2
indices = np.arange(n_stations)
# green_colors = ['#6AA84F', '#38761D', '#274E13', '#3C6']
# green_colors = ['#00a77d',  '#007c5e', '#004935', '#333333']
# Plotting each metric
def add_labels(ax, bars, fontsize):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.2f}'.format(height),
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 9),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom',
        fontsize=fontsize)
        
for i, metric in enumerate(['MSE', 'MAE', 'RMSE', 'R2_score']):
    bars = ax.bar(indices + i * bar_width,
           df_output[metric],
           width=bar_width,
           label=metric,
           # color=green_colors[i % len(green_colors)]
           )
    add_labels(ax, bars, fontsize=16)

# Add labels, title and axes ticks
ax.set_xlabel('Station')
ax.set_ylabel('Value')
ax.set_ylim(0, 23)
ax.set_title('Metrics by Station')
ax.set_xticks(indices + bar_width * 1.5)
ax.set_xticklabels(df_output['station'])
# ax.legend()

# Create a secondary y-axis for the distance line graph
ax2 = ax.twinx()

# Plotting the distance as a line graph
line, = ax2.plot(df_output['station'],
                 df_output['displacement'],
                 color='darkgreen',
                 linewidth=3,
                 marker='o',
                 label='Displacement')

# Add labels for the line graph
ax2.set_ylabel('Displacement', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize)

# Adjust the y-axis range and appearance if needed
ax2.set_ylim(0, max(df_output['displacement']) + 50)

# Add legend for both plots
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# Save and show plot
# plt.savefig('four-external-validated-forecasts.png')
plt.show()


