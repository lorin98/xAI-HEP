from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def build_cnn(height: int = 9,
              width: int = 384,
              depth: int = 1):
  
    inputs = Input(shape=(height, width, depth))

    x = Conv2D(16, (5, 5), padding='same', activation='relu')(inputs)
    #x = BatchNormalization()(x)

    x = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
    #x = BatchNormalization()(x)

    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    #x = BatchNormalization()(x)

    x = Conv2D(128, (5, 5), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x) # IMPORTANT for RAM!!!

    x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.3)(x)

    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)

    x = Dense(2, activation='linear')(x) # LINEAR activation fn for regression

    model = Model(inputs, x)
    return model

def build_cnn_tracin(height: int = 9,
              width: int = 384,
              depth: int = 1):
  
    inputs = Input(shape=(height, width, depth))

    x = Conv2D(16, (5, 5), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = Conv2D(128, (5, 5), padding='same', activation='relu')(x)

    x = GlobalAveragePooling2D()(x) # IMPORTANT for RAM!!!
    
    x = Dense(2, activation='linear')(x) # LINEAR activation fn for regression

    model = Model(inputs, x)
    return model