#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:05:25 2023

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
import mymodels
import keras

#%https://www.kaggle.com/code/rakshitacharya/convlstm1d-for-weather-data
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#%
def categorize(pm_list):
    pm_cat = []
    for pm in pm_list:
        if pm >= 0 and pm <= 15.0:
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
df = pd.read_csv('pm2013_2023.csv')

n = len(df)
print(n)

pm = df['PM'].values
plt.figure(dpi=300)
plt.hist(pm, bins='auto')
plt.title('pm')
#% pm=>rescaled
sc = MinMaxScaler()
pm_rescaled_1 = sc.fit_transform(pm.reshape(-1,1))
init_lr = 10e-4
epochs = 50
batch_size = 2
model_name = 'Model-4'
# model_name = 'Model-1'
df_output = pd.DataFrame([])
df_slice = pd.DataFrame([])
time_series_data = np.array(pm_rescaled_1)  
# W_ins = [7, 7, 7, 8, 15]  # The input window size from your table
# W_outs = [1, 3, 7, 2, 5]  # The output range from your table
W_ins = [7]
W_outs = [1]
# Strides = [1, 2, 3, 4, 5, 6, 7]
Strides = [6]

#%%
for stride in Strides:   
    for W_in, W_out in zip(W_ins, W_outs):
        data_note = f'pm-{W_in}-{W_out}-{stride}-bs{batch_size}'  
        num_classes = W_out
        print(data_note)
        
        X, y = create_dataset_with_stride(time_series_data, W_in, W_out, stride)
        # X_pm, y_pm = create_dataset_with_stride(pm, W_in, W_out, stride)
    
        #%
        # Save to CSV
        # np.savetxt("PM-input-7-1-6.csv", X.flatten(), delimiter=",",  fmt='%.4f')
        # np.savetxt("PM-output-7-1-6.csv", y.flatten(), delimiter=",",  fmt='%.4f')
        
        # fmt='%d' is used for integer formatting. For floating-point numbers, you can use fmt='%.5f' for 5 decimal places, for example.
        
        X = np.squeeze(X)
        y = np.squeeze(y)
        
        train_len = int(len(X)*0.8)
        X_train = X[:train_len]
        X_test = X[train_len:]
        # X_test_pm = X_pm[train_len: ]    
        
        train_len2 = int(len(y)*0.8)
        y_train = y[:train_len2]
        y_test = y[train_len2:]
        # y_test_pm = y_pm[train_len2:]
        
        df_slice = df_slice._append(pd.DataFrame(
            {
            "W_in": [W_in],
            "W_out": [W_out],
            "stride": [stride],
            "X.shape": [X.shape],
            "y.shape": [y.shape],
            "X_train": [X_train.shape],
            "X_test": [X_train.shape],
            "y_train": [y_train.shape],
            "y_test": [y_train.shape],
            },
            index=[0]),
            ignore_index=True)
        print(df_slice)
    
    #%%
    X_train = np.reshape(X_train, 
                    (X_train.shape[0],
                    1,
                    X_train.shape[1], 1))
    
    print(X_train.shape) #(2395, 1, 15, 1)
    
    X_test = np.reshape(X_test, 
                    (X_test.shape[0],
                    1,
                    X_test.shape[1],
                    1)
                    )
    print(X_test.shape, y_test.shape)
    
    if W_out==1:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    print(y_train.shape, y_test.shape)
        
    init_lr = 10e-4
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_loss',
        patience = 2,
        verbose=1,
        factor=0.75,
        min_lr=0.00001)
    
    early_stop = EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                    min_delta=init_lr/100)
    
    #%%   
    inp = layers.Input(shape=(1, W_in, 1))
    print(inp.shape)   
    
    if model_name == 'Model-1':
        model = mymodels.build_model_1(inp, num_classes)  
    elif model_name == 'Model-5':
        model = mymodels.build_model_5(inp, num_classes)    
    else:
        model = mymodels.build_model_4(inp, num_classes)  
    
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate = init_lr),
        # optimizer='adam',
        loss='mse',
        # loss='mae',
        metrics=['mae', 'mse'])
    
    print(model.summary())
      
    backup_model_best = f'{model.name}-{data_note}-50ep.hdf5'
    print('\nbackup_model_best: ', backup_model_best)
    mcp2 = ModelCheckpoint(
        backup_model_best,
        save_best_only=True)
    # Plotting the model
    # model_diagram = keras.utils.plot_model(model, 
    #     show_shapes=True, 
    #     show_layer_names=True,
    #     to_file=f"{model_name}.png")
    # print('Saved model: ', f"{model_name}.png")

    
    #%%
    t3 = datetime.now()
    history1 = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        # epochs=1,
        batch_size=batch_size,
        validation_split=0.3,
        # validation_data=(x_valid, y_valid),
        verbose=1,
        callbacks=[
            # learning_rate_reduction,
            # early_stop,
            mcp2,
            tqdm_callback,
            PlotLossesKeras(),
            ]
        )
    
    t4 = datetime.now() - t3
    print("\nTraining time: ", t4)

    
    #%%
    history = history1
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("LOSS")
    plt.legend(["TRAIN" , "TEST"])
    plt.show()
    
    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.title("MAE")
    plt.legend(["TRAIN" , "TEST"])
    plt.show()
    
    plt.plot(history.history["mse"])
    plt.plot(history.history["val_mse"])
    plt.title("MSE")
    plt.legend(["TRAIN" , "TEST"])
    plt.show()
    
    #%%
    from tensorflow.keras.models import load_model
    # 7-1-6 model
    # backup_model_best ='Model-ConvLSTM1D-BiLSTM-pm-7-1-6-bs2-loss-mse-50ep-20230117.hdf5'
    model = load_model(backup_model_best)
    print(model.summary())
    print('Loaded: ', backup_model_best)
        
    #%%
    y_pred = model.predict(X_test, verbose=1)
    
    #%%
    print("Predict PM values")
    y_test_reversed = sc.inverse_transform(y_test.reshape(-1,1))
    y_pred_reversed = sc.inverse_transform(y_pred.reshape(-1,1))
    
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
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(y_test_reversed.flatten(), color='blue')
    plt.plot(y_pred_reversed.flatten(), color='orange')
    plt.legend(["actual", "pred"])
    plt.title("Prediction of PM " + f"{model.name}-{data_note}")
    plt.xlabel(f"MAE: {mae:.2f}")
    # plt.savefig(f"{model.name}-Hat-Yai.png")
    plt.show()
    
    #%%
    df_output = df_output._append(pd.DataFrame(
        {
        "W_in": [W_in],
        "W_out": [W_out],
        "stride": [stride],
        "MSE": [mse],
        "MAE": [mae],
        "RMSE": [rmse],
        "R2_score": [r2]
        },
        index=[0]),
        ignore_index=True)
    print(df_output)
            
#%%   
# df_output_file = f"{model.name}-{data_note}-tune.csv"
# df_output.to_csv(df_output_file)

#%%
print("Results in classification task")

y_test_cat = categorize(y_test_reversed)
y_pred_cat = categorize(y_pred_reversed)

np.unique(y_test_cat, return_counts=True)
pred_cat, pred_cnt = np.unique(y_pred_cat, return_counts=True)

from sklearn.metrics import classification_report
report = classification_report(y_test_cat, y_pred_cat)
print(report)

#%% print confuse matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate the normalized confusion matrix
cm = confusion_matrix(y_test_cat,
                      y_pred_cat,
                      normalize='true')

# Set the DPI and font size for the plot
dpi = 300
fontsize = 16
plt.figure(figsize=(8, 6), dpi=dpi)
sns.set(font_scale=1.5)

# Create the heatmap for the normalized confusion matrix
ax = sns.heatmap(cm, annot=True, 
                 fmt='.2f',
                 cmap='Greens',
                 cbar=False)

# Setting the labels and title
ax.set_xlabel('Predicted PM2.5 Level', fontsize=fontsize)
ax.set_ylabel('True PM2.5 Level', fontsize=fontsize)
# ax.set_title('Normalized Confusion Matrix', fontsize=fontsize)

# Ticks
ax.set_xticklabels(pred_cat, fontsize=fontsize)
ax.set_yticklabels(pred_cat, fontsize=fontsize)

plt.tight_layout()
# plt.savefig("X-test-confusion-matrix.png")
plt.show()
