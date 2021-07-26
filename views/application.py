import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from plotly.subplots import make_subplots
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def write():
    sns.set_theme(style="whitegrid")

    header = st.beta_container()
    dataset = st.beta_container()
    features = st.beta_container()
    model_training = st.beta_container()


    with header:
        st.title('Application de la PCA')
        st.text('In this first data science project with streamlit, I\'ll: \n\t1-\tInterduce pca model,\n\t2-\tCreate it,\n\t3-\tFit it \n\t4-\tAnd deploy it.\n ...')

        
    with dataset:
        st.header('Dataset Section')
        st.text('In this section, I will use digit dataset from sklearn library to my PCA model ...')
        digit_data = load_digits()
        k = digit_data.keys()
        n = random.randint(0, 1796)
        nrows = 4
        ncols = 4
        fig = plt.gcf()
        fig.set_size_inches(ncols * 4, nrows * 4)
        for i, img in enumerate(digit_data.images[n:n+8]):
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off') 
            plt.imshow(img.reshape([8, 8]))
        plt.show()
        st.write(fig)
    
        dataset_expander = st.beta_expander(label='Show the original digit dataset !')
    with dataset_expander:
        df = pd.DataFrame(digit_data.data, columns=digit_data.feature_names)
        df['Target'] = pd.DataFrame(digit_data.target)
        st.text('Show the original Dataset:')
        st.write(df)
        st.text('Shape of original Dataset:')
        st.write(df.shape)
        st.text('Show the unique values of our target:')
        labels = df['Target'].unique()
        st.write(labels)

    with features:
        st.header('Features Section')
        features_expander = st.beta_expander(label='Show features !')
        with features_expander:
            st.text('In this section, I will prepare my features for the model fitting...')
            X, y = digit_data.data, digit_data.target
            st.write(X[0, :].reshape([8, 8]))


    
    with model_training:
        st.header('Model Section')
        st.text('In this section, I will: \n\t1-\tImplement the PCA model, \n\t2-\tPlot and Analyse it, \n\t3-\tAnd deploy it using sklearn library ...')
        st.text('-----------------------------------------------------------------------\n')
        
        model1_expander = st.beta_expander(label='First PCA model with 2 components!')
        with model1_expander:
            st.text('\n\t\t=====> Apply our first PCA model with 2 components!\n')
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(X)
            pca1 = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
            pcadf1 = pd.concat([pca1, df[['Target']]], axis = 1)
            st.write(pcadf1.head(10))
            st.text('Shape of the dataset after being reduced to 2 features:')
            st.write(pca1.shape)

            st.text('\n\t\t=====> Plot the two dimensions!\n')
            fig2, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(pca1['principal component 1'], 
                                pca1['principal component 2'], 
                                c = df['Target'], alpha=0.8)
            plt.title('Plotting the 2-Dimensional data after PCA is applied', fontsize = 20)
    #         plt.xlim(-30, 30)
    #         plt.ylim(-30, 30)
            plt.xlabel('Principal Component 1', fontsize = 15)
            plt.ylabel('Principal Component 2', fontsize = 15)
            plt.legend(*scatter.legend_elements(), loc="best", title="Classes")
            for i in range(797):
                plt.text(principalComponents[i,0], principalComponents[i,1], str(y[i]))
            ax.plot([])
            ax.grid()
            plt.show()
            st.write(fig2)
            
            st.text('Comment this graph: \n\t ...')
        
    model2_expander = st.beta_expander(label='Second PCA model with 3 components!')
    with model2_expander:   
        st.text('\n\t\t=====> Apply our first PCA model with 3 components!\n')
        pca_ = PCA(n_components=3)
        principalComponents_ = pca_.fit_transform(X)
        pca2 = pd.DataFrame(data = principalComponents_, columns = ['principal component 1',
                                                                    'principal component 2',
                                                                    'principal component 3'])
        st.write(pca2.head(10))
        st.text('Shape of the dataset after being reduced to 3 features:')
        st.write(pca2.shape)

        st.text('\n\t\t=====> Plot the three dimensions!\n')
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        scatter3 = ax3.scatter(pca2['principal component 1'], 
                             pca2['principal component 2'], 
                             c = pca2['principal component 3'] , 
                             s = df['Target'],
                            #cmap = 'Blues'
                              )
        plt.title('Plotting the 3-Dimensional data after PCA is applied', fontsize = 20)
        plt.xlabel('Principal Component 1', fontsize = 15)
        plt.ylabel('Principal Component 2', fontsize = 15)
        legend1 = ax3.legend(*scatter3.legend_elements(),
                            loc="lower left", title="Principal\nComponent 3")
        ax3.add_artist(legend1)
        handles, labels = scatter3.legend_elements(prop="sizes", alpha=0.6)
        legend2 = ax3.legend(handles, labels, loc="upper right", title="Classes")
        ax.grid()
        plt.show()
        st.write(fig3)

        st.text('\n\t\t=====> Plot the combinations groups of 2 components out of 3 after applying PCA!\n')

        fig, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(24, 8))
        plt.suptitle("Plotting combinations of all 3 components after applying PCA", fontsize = 25)

        ax[0].scatter(pca2['principal component 1'], 
                      pca2['principal component 2'], 
                      c = df['Target'], alpha=0.8, 
                      cmap = "inferno")
        ax[0].set_xlabel('Principal Component 1', fontsize = 15)
        ax[0].set_ylabel('Principal Component 2', fontsize = 15)
        ax[0].legend(*scatter.legend_elements(), loc="best", title="Classes")
        ax[0].grid()

        scatter = ax[1].scatter(pca2['principal component 1'], 
                                pca2['principal component 3'], 
                                c = df['Target'], alpha=0.8, 
                                cmap = "inferno")
        ax[1].set_xlabel('Principal Component 1', fontsize = 15)
        ax[1].set_ylabel('Principal Component 3', fontsize = 15)
        ax[1].legend(*scatter.legend_elements(), loc="best", title="Classes")
        ax[1].grid()

        scatter = ax[2].scatter(pca2['principal component 2'], 
                                pca2['principal component 3'], 
                                c = df['Target'], alpha=0.8, 
                                cmap = "inferno")
        ax[2].set_xlabel('Principal Component 2', fontsize = 15)
        ax[2].set_ylabel('Principal Component 3', fontsize = 15)
        ax[2].legend(*scatter.legend_elements(), loc="best", title="Classes")
        ax[2].grid()

        plt.show()
        st.write(fig)
        
    
    model3_expander = st.beta_expander(label='Third PCA model ')
    with model3_expander:
        n = st.number_input('Enter a number')
        col1, col2 = st.beta_columns(2)
        original = Image.open('./img/use_my_voice.jpg')
        col1.header("Original")
        col1.image(original, use_column_width=True)
        
        # Loading the image with cv2 
        img = cv2.imread('./img/use_my_voice.jpg')
        # Splitting the image in R,G,B arrays.
        blue,green,red = cv2.split(img) 
        #it will split the original image into Blue, Green and Red arrays.

        #initialize PCA with first 20 principal components
        pca = PCA(n_components=int(n))

        #Applying to red channel and then applying inverse transform to transformed array.
        red_transformed = pca.fit_transform(red)
        red_inverted = pca.inverse_transform(red_transformed)

        #Applying to Green channel and then applying inverse transform to transformed array.
        green_transformed = pca.fit_transform(green)
        green_inverted = pca.inverse_transform(green_transformed)

        #Applying to Blue channel and then applying inverse transform to transformed array.
        blue_transformed = pca.fit_transform(blue)
        blue_inverted = pca.inverse_transform(blue_transformed)

        img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)
        
        col2.header("PCAscale")
        col2.image(img_compressed, use_column_width=True)