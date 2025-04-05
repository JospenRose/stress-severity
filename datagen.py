import numpy as np
import pandas as pd
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import seaborn as sns
from save_load import save
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_extraction import statistical_features, image_feature_extractor
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def plot_image(image, denoised_image, img_equalized, i):
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image', fontweight='bold', fontfamily='Serif')
    plt.axis(False)

    plt.subplot(132)
    plt.imshow(denoised_image)
    plt.title('Denoised Image', fontweight='bold', fontfamily='Serif')
    plt.axis(False)

    plt.subplot(133)
    plt.imshow(img_equalized)
    plt.title('Equalized Image', fontweight='bold', fontfamily='Serif')
    plt.axis(False)

    plt.savefig(f'Data visualization/image {i}.png')
    plt.close()


def datagen():
    data = pd.read_excel('data/data.xlsx')

    # Data cleaning
    data.dropna(inplace=True)

    image_features = []
    index_to_remove = []

    img_feature_extractor = image_feature_extractor()

    for i, row in data.iterrows():
        image_path = row['Image Path']

        image = cv2.imread(f'{image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            index_to_remove.append(i)
            continue

        # Image preprocessing
        # resize image
        image = cv2.resize(image, (224, 224))
        # Denoising
        denoised_image = cv2.fastNlMeansDenoisingColored(image)
        # histogram equalization
        img_equalized = exposure.equalize_adapthist(denoised_image)

        # if i <= 4:
            # plot_image(image, denoised_image, img_equalized, i)

        # feature extraction - extract shallow, intermediate and deep features from EfficientNet
        preprocessed_image = np.expand_dims(img_equalized, axis=0)
        preprocessed_image = preprocess_input(preprocessed_image)  # Normalize for EfficientNet

        shallow, intermediate, deep = img_feature_extractor.predict(preprocessed_image)
        shallow_feat = np.argmax(shallow, axis=1).argmax(axis=1).squeeze()
        intermediate_feat = np.argmax(intermediate, axis=1).argmax(axis=1).squeeze()
        deep_feat = np.argmax(deep, axis=1).argmax(axis=1).squeeze()

        img_feature = np.concatenate([shallow_feat, intermediate_feat, deep_feat])

        image_features.append(img_feature)

    physical_data = data.drop(columns=['Image Path'], index=index_to_remove)
    label = physical_data['Stress Level']
    label = np.array(label)
    physical_data = physical_data.drop(columns=['Stress Level'])

    # Compute the correlation matrix
    correlation_matrix = physical_data.corr()

    # Create a heatmap
    plt.figure(figsize=(12, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)

    plt.title("Correlation Heatmap of Physical Data", fontweight='bold', fontfamily='Serif')
    plt.xticks(fontweight='bold', fontfamily='Serif')
    plt.yticks(fontweight='bold', fontfamily='Serif')
    plt.savefig('Data visualization/Correlation - physical data.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=physical_data, palette="muted", inner="quartile")
    plt.title("Physical Data", fontweight='bold', fontfamily='Serif')
    plt.xlabel("Features", fontweight='bold', fontfamily='Serif')
    plt.ylabel("Values", fontweight='bold', fontfamily='Serif')
    plt.xticks(fontweight='bold', fontfamily='Serif')
    plt.yticks(fontweight='bold', fontfamily='Serif')
    plt.savefig('Data visualization/physical data.png')
    plt.close()

    # Physical data preprocessing
    # Normalization - Z-score
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(physical_data)
    normalized_data = pd.DataFrame(normalized_data, columns=physical_data.columns)

    # visualize normalized data
    pairplot = sns.pairplot(normalized_data, diag_kind="kde")

    pairplot.fig.suptitle("Pairwise Relationships of Normalized Physical Data",
                          y=1.02, fontweight='bold', fontfamily='Serif')

    for ax in pairplot.axes.flatten():
        ax.tick_params(axis='x', labelsize=10, width=2, direction='in', length=6,
                       grid_color='black', grid_alpha=0.5)
        ax.tick_params(axis='y', labelsize=10, width=2, direction='in', length=6,
                       grid_color='black', grid_alpha=0.5)
        ax.xaxis.label.set_fontfamily('serif')
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontfamily('serif')
        ax.yaxis.label.set_fontweight('bold')

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily('serif')
            tick.set_fontweight('bold')

    pairplot.savefig('Data visualization/normalized-physical_data.png')
    plt.close()

    statistical_feat = normalized_data.apply(statistical_features, axis=1)
    statistical_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    statistical_feat.plot(kind='box', vert=False, patch_artist=True)
    plt.title('Statistical Features', fontweight='bold', fontfamily='Serif')
    plt.xlabel('Value', fontweight='bold', fontfamily='Serif')
    plt.ylabel('Statistical Features', fontweight='bold', fontfamily='Serif')
    plt.xticks(fontweight='bold', fontfamily='Serif')
    plt.yticks(fontweight='bold', fontfamily='Serif')
    plt.savefig('Data visualization/statistical features.png')
    plt.close()

    statistical_feat = np.array(statistical_feat)

    image_features = np.array(image_features)

    # feature fusion - (Attention based weighted fusion) (EWFI-FUSION NET)

    statistical_feat = tf.convert_to_tensor(statistical_feat, dtype=tf.float32)
    img_features = tf.convert_to_tensor(image_features, dtype=tf.float32)

    # Compute attention scores
    statistical_weights = tf.keras.layers.Dense(1, activation="softmax")(statistical_feat)
    image_weights = tf.keras.layers.Dense(1, activation="softmax")(img_features)

    # Weighted sum of features
    weighted_statistical_feat = statistical_weights * statistical_feat
    weighted_image_features = image_weights * img_features

    # Combine the weighted features
    fused_features = tf.concat([weighted_statistical_feat, weighted_image_features], axis=1)
    fused_features = fused_features.numpy()
    fused_features = np.nan_to_num(fused_features)

    # Select the 500 best features using ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=61)
    selected_features = selector.fit_transform(fused_features, label)

    # selected_feature_indices = selector.get_support(indices=True)
    # print(f"Shape of selected features: {selected_features.shape}")
    # print(f"Selected feature indices: {selected_feature_indices}")

    # Absolute
    selected_features = np.abs(selected_features)
    # Normalized features
    selected_features = selected_features / np.max(selected_features, axis=0)
    # Nan to Num conversion
    selected_features = np.nan_to_num(selected_features)

    # training testing split
    train_sizes = [0.7, 0.8]
    for train_size in train_sizes:
        x_train, x_test, y_train, y_test = train_test_split(selected_features, label, train_size=train_size)
        save('x_train_' + str(int(train_size * 100)), x_train)
        save('y_train_' + str(int(train_size * 100)), y_train)
        save('x_test_' + str(int(train_size * 100)), x_test)
        save('y_test_' + str(int(train_size * 100)), y_test)
