from pca_img.import_lib import *
from pca_img.transform_data import *

def train_test_shape_labels(x_train, x_test, y_train, y_test):
    classes = np.unique(y_train)
    nClasses = len(classes)
    return print(f'Traning X data shape: {x_train.shape}\nTesting X data shape: {x_test.shape}\n****************\nTraning y data shape: {y_train.shape}\nTesting y data shape: {y_test.shape}\n\n****************\n\nTotal number of outputs : {nClasses}\nOutput classes : {classes}')


def viz_img(x_data, y_data, index_1, index_2):
    plt.figure(figsize=[5,5])
    # Display the first image in training data
    plt.subplot(121)
    curr_img = np.reshape(x_data[index_1], (32,32,3))
    plt.imshow(curr_img) 
    return print(plt.title("(Label: " + str(create_labels()[y_data[index_1][index_2]]) + ")"))

def min_max_img(x_train):
    x_train_1 = x_train/255.0
    print(f'Min and max values : {np.min(x_train),np.max(x_train)}\nMin and max values must be between 0 and 1 : {np.min(x_train_1)}, {np.max(x_train_1)}\n{x_train_1.shape}')

def scatter_component(df):
    plt.figure(figsize=(16,10))
    return sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )