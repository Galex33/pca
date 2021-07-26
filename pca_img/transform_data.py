from pca_img.import_lib import *

def train_test():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = train_test()

def create_labels():
    label_dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
    }
    return label_dict

def applause_img(x_train):
    x_train = x_train/255.0
    x_train_flat = x_train.reshape(-1,3072)
    feat_cols = ['pixel'+ str(i) for i in range(x_train_flat.shape[1])]
    return feat_cols

def create_df(x_train, y_train):
    feat_cols= applause_img(x_train)
    x_train = x_train/255.0
    x_train_flat = x_train.reshape(-1,3072)
    df_cifar = pd.DataFrame(x_train_flat,columns=applause_img(x_train))
    df_cifar['label'] = y_train
    print('Size of the dataframe: {}'.format(df_cifar.shape))
    return df_cifar

def pca_apply(x_train, y_train, n_c):
    df_cifar = create_df(x_train, y_train)
    pca_cifar = PCA(n_components=n_c)
    principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:,:-1])
    principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar, columns = ['principal component 1', 'principal component 2'])
    principal_cifar_Df['y'] = y_train
    print('Explained variation per principal component: {}'.format(pca_cifar.explained_variance_ratio_))
    return principal_cifar_Df

def normalize_test(x_test):
    x_test = x_test/255.0
    x_test = x_test.reshape(-1,32,32,3)
    x_test_flat = x_test.reshape(-1,3072)
    return x_test_flat

def pca_percentage(x_train_flat, x_test_flat, y_train, y_test, pca):
    # Quantité de variance explicité, remplace le nbr de composant (n_components)
    pca = PCA(pca)
    pca.fit(x_train_flat)
    # Nbr de composant pour atteindre 90% de variance
    print(f'number of components for 90% of the total : {pca.n_components_}')
    train_img_pca = pca.transform(x_train_flat)
    test_img_pca = pca.transform(x_test_flat)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return y_train, y_test, train_img_pca, test_img_pca

def picture_compress(photo):
    img = np.array(photo)
    x_test = img/255.0
    x_test = x_test.reshape(-1,32,32,3)
    x_test_flat = x_test.reshape(-1,3072)
    # Quantité de variance explicité, remplace le nbr de composant (n_components)
    pca = PCA(0.9)
    pca.fit(x_test_flat)
    photo_compress = pca.transform(img)
    return photo_compress, x_test_flat, 