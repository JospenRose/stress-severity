import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Concatenate, Add, Conv2D


def statistical_features(row):
    return pd.Series({
        'mean': row.mean(),
        'median': row.median(),
        'std': row.std(),
        'min': row.min(),
        'max': row.max(),
        'variance': row.var(),
        'skew': skew(row),
        'kurtosis': kurtosis(row),
        'entropy' : entropy(row)
    })


def image_feature_extractor():

    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Extract specific layers for shallow, intermediate, and deep features
    shallow_layer = base_model.get_layer('block1a_project_bn').output
    intermediate_layer = base_model.get_layer('block3b_add').output
    deep_layer = base_model.get_layer('top_activation').output

    # Cascade Connection: Upsample all layers to the same resolution and concatenate them
    shallow_upsampled = UpSampling2D(size=(2, 2))(shallow_layer)
    intermediate_upsampled = UpSampling2D(size=(8, 8))(intermediate_layer)
    deep_upsampled = UpSampling2D(size=(32, 32))(deep_layer)

    cascade_connection = Concatenate(axis=-1)([shallow_upsampled, intermediate_upsampled, deep_upsampled])

    # Residual Connection: Add shallow layer to intermediate layer
    aligned_intermediate_layer = UpSampling2D(size=(4, 4))(intermediate_layer)
    aligned_intermediate_layer = Conv2D(filters=16, kernel_size=1, padding='same')(aligned_intermediate_layer)
    residual_connection = Add()([shallow_layer, aligned_intermediate_layer])

    # Upsample deep_layer to match shallow_layer's shape
    upsampled_deep_layer = UpSampling2D(size=(16, 16))(deep_layer)
    upsampled_deep_layer = Conv2D(filters=16, kernel_size=1, padding='same')(upsampled_deep_layer)

    # Concatenate all layers to form dense connection
    dense_connection = Concatenate()([shallow_layer, aligned_intermediate_layer, upsampled_deep_layer])

    # Feature extraction with cascade and residual connections
    feature_extractor = Model(inputs=base_model.input, outputs=[cascade_connection, residual_connection, dense_connection])

    return feature_extractor