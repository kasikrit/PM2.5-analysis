#%%
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend, optimizers

def build_model_1(input_shape, num_classes, activation='linear'):
    inp = input_shape
    print('In model, ', inp.shape)
    x = layers.ConvLSTM1D( #first tf 2.6
        filters=64,
        kernel_size=(7),
        padding="same",
        return_sequences=True,
        activation="swish",
    )(inp)
    print(x.shape)
    conv = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(conv)
    
    #   x = layers.BatchNormalization()(x) #misstake
    x = layers.ConvLSTM1D(
        filters=128,
        kernel_size=(3),
        padding="same",
        return_sequences=True,
        activation="swish",
    )(x)
    conv = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(x)
    print(x.shape) #(None, 1, 15, 64)
    
    output_reshaped = layers.Reshape((1, 1, inp.shape[2], 128))(x)
    # Define Conv3D layer
    conv3d_layer = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), 
                          activation='swish', padding='same')
    # Apply Conv3D to reshaped output data
    conv3d_output = conv3d_layer(output_reshaped)
    print(conv3d_output.shape)
    
    x = layers.Dropout(0.3)(conv3d_output)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, 
                     activation='swish',
                     )(x)
    x = layers.Dense(num_classes,
                     activation=activation,
                     )(x)
    
    model = keras.models.Model(inp, x, name='Model-ConvLSTM1D')
    model.summary()  
    
    return model

#%%
def build_model_2(input_shape, num_classes, activation='linear'):
    inp = keras.Input(shape=input_shape)   
    x = layers.LSTM(
        units=32,
        return_sequences=True,
        activation="swish",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(
        units=64,
        return_sequences=True,
        activation="swish",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(
        units=128,
        return_sequences=False,
        activation="swish",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Reshape is no longer necessary as LSTM outputs 2D tensor
    # Flatten the output to make it suitable for dense layers
    x = layers.Flatten()(x)
    
    # Dense layers
    x = layers.Dense(64, activation='swish')(x)
    x = layers.Dense(num_classes,
                     activation=activation,
                     )(x)
    
    # Create the model
    model = keras.Model(inputs=inp, outputs=x, name='Model-LSTM')
    model.summary()
    
    return model

#%%
def build_model_3(input_shape, num_classes):
    inp = input_shape
    print('In model, ', inp.shape)
    x = layers.ConvLSTM1D( #first tf 2.6
        filters=32,
        kernel_size=(7),
        padding="same",
        return_sequences=True,
        activation="swish",
    )(inp)
    print(x.shape)
    conv = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(conv)
    
    x = layers.ConvLSTM1D( #first tf 2.6
        filters=64,
        kernel_size=(7),
        padding="same",
        return_sequences=True,
        activation="swish",
    )(x)
    print(x.shape)
    conv = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(conv)
    
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM1D(
        filters=128,
        kernel_size=(3),
        padding="same",
        return_sequences=True,
        activation="swish",
    )(x)
    conv = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(x)
    print(x.shape) #(None, 1, 15, 64)
    
    output_reshaped = layers.Reshape((1, 1, inp.shape[2], 128))(x)
    # Define Conv3D layer
    conv3d_layer = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), 
                          activation='swish', padding='same')
    # Apply Conv3D to reshaped output data
    conv3d_output = conv3d_layer(output_reshaped)
    print(conv3d_output.shape)
    
    x = layers.Dropout(0.3)(conv3d_output)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(128, 
                     activation='swish',
                     )(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, 
                     activation='swish',
                     )(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, 
                     activation='swish',
                     )(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(num_classes,
                     activation='linear',
                     )(x)
    
    model = keras.models.Model(inp, x, name='Model-3-ConvLSTM1D')
    model.summary()  
    
    return model

#%%
from keras.layers import Bidirectional, LSTM
# from keras import backend as K
# from keras.layers import Lambda

# ConvLSTM1D + 3D CNN + BiLSTM + Dense
def build_model_4(input_shape,
                  num_classes, 
                  activation='linear'):
    inp = input_shape
    print(f"{inp.shape=}")
    x = layers.ConvLSTM1D(
        filters=64,
        kernel_size=7,
        padding="same",
        return_sequences=True,
        activation="swish",
    )(inp)
    
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(x)
    print(x.shape)
    
    x = layers.ConvLSTM1D(
        filters=128,
        kernel_size=3,
        padding="same",
        return_sequences=True,
        activation="swish",
    )(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(x)
    print(x.shape)
    
    output_reshaped = layers.Reshape((1, 1, inp.shape[2], 128))(x)
    
    # output_reshaped = Lambda(lambda t: K.reshape(t, (-1, 1, 1, input_shape[2], 128)))(x)
    
    conv3d_output = layers.Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        activation='swish',
        padding='same'
    )(output_reshaped)
    print(output_reshaped)

    # Reshape or use TimeDistributed here if needed (depends on your data)
    # For example:
    # x = layers.Reshape(target_shape)(conv3d_output)
    # or
    x = layers.Dropout(0.3)(conv3d_output)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # BiLSTM Layer
    x = Bidirectional(LSTM(
        units=64,
        # units=32,
        return_sequences=False))(x)

    # Output Layer
    x = layers.Dense(num_classes,
            activation=activation)(x)
    
    model = keras.models.Model(inp, x,
                    name='Model-ConvLSTM1D-BiLSTM')
    return model

#%%
def build_model_5(input_shape,
                  num_classes, 
                  activation='linear'):
    inp = input_shape
    print(f"{inp.shape=}")
    x = layers.ConvLSTM1D(
        filters=64,
        kernel_size=7,
        padding="same",
        return_sequences=True,
        activation="swish",
    )(inp)
    
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(x)
    print(x.shape)
    
    x = layers.ConvLSTM1D(
        filters=128,
        kernel_size=3,
        padding="same",
        return_sequences=True,
        activation="swish",
    )(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.3)(x)
    print(x.shape)
    
    output_reshaped = layers.Reshape((1, 1, inp.shape[2], 128))(x)
    
    # output_reshaped = Lambda(lambda t: K.reshape(t, (-1, 1, 1, input_shape[2], 128)))(x)
    
    conv3d_output = layers.Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        activation='swish',
        padding='same'
    )(output_reshaped)
    print(output_reshaped)

    # Reshape or use TimeDistributed here if needed (depends on your data)
    # For example:
    # x = layers.Reshape(target_shape)(conv3d_output)
    # or
    x = layers.Dropout(0.3)(conv3d_output)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # BiLSTM Layer
    x = Bidirectional(LSTM(
        #units=64,
        units=32,
        return_sequences=True))(x)
    
    x = Bidirectional(LSTM(
        #units=64,
        units=16,
        return_sequences=True))(x)
    
    x = Bidirectional(LSTM(
        #units=64,
        units=8,
        return_sequences=False))(x)

    # Output Layer
    x = layers.Dense(num_classes,
            activation=activation)(x)
    
    model = keras.models.Model(inp, x,
                    name='Model-ConvLSTM1D-3BiLSTM')
    return model





