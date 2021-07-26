import streamlit as st
from pca_img.viz import *
from pca_img.transform_data import *
from pca_img.model_dl import *

def write():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    principal_cifar_Df = pca_apply(x_train, y_train, 2)
    def scatter_component(df):
        sns.scatterplot(
            x="principal component 1", y="principal component 2",
            hue="y",
            palette=sns.color_palette("hls", 10),
            data=df,
            legend="full",
            alpha=0.3
        )
        return st.pyplot()

    def viz_img(x_data, y_data, index_1, index_2):
        # Display the first image in training data
        # st.pyplot.subplot(121)
        attention_array = np.reshape(x_data[index_1], (32,32,3))
        fig, ax = plt.subplots()
        im = ax.imshow(attention_array)
        st.pyplot()
      
    
    st.title("Visualisation")

    st.write("Les données qui seront ici manipulées sont issues des données libre d'accsè de Keras. La multidimensionalité exprimé par ce jeu de donné va permettre d'exprimer pleinement l'interet de la PCA")
    st.write("Le jeu de Traning X a pour dimensions : (50000, 32, 32, 3)")
    st.write("Le jeu de Test X a pour dimensions : (10000, 32, 32, 3)")
    st.write("****************")
    st.write("Le jeu Traning y a pour dimensions : (50000, 1)")
    st.write("Le jeu Testing y a pour dimensions : (10000, 1)")
    st.write("****************")
    st.write("Le nombre de classes est de 10 et organisées sous ce format : [0 1 2 3 4 5 6 7 8 9]")


    st.header("Représentation visuel des données")
    st.subheader("Label: Frog")
    viz_img(x_train, y_train, 0, 0), 
    st.subheader("Label: Cat")
    viz_img(x_test, y_test, 0, 0)
    st.write("Les deux photos sont issus du jeu de train et de test. La qualité est de 32 pixels sur 32 pixels")


    st.header("Scatterplot avec deux composants principaux")
    st.write("Nous allons ici soumettre les données du jeu de train et le réduire à deux composant principaux")
    st.write("Le dataframe ")
    st.write("La variation que chaque composant explique est de 0.2907663 et pous le second 0.11253144")
    st.write("La quantité d'informations par les composants principaux 1 et 2 est décente , passé de 3072 dimension a 2 composants principaux. Cependant, il y a un certain chevauchement sématique de classes  dans cet ensemble de données, une grenouille peut avoir une forme légèrement similaire à celle d'un chat ou d'un cerf avec un chien ; surtout lorsqu'il est projeté dans un espace à deux dimensions.")
    scatter_component(principal_cifar_Df)
    st.write("Les points appartenant à une même classe sont proches les uns des autres, et les points ou images sémantiquement très différents sont plus éloignés les uns des autres.")

    # y_train, y_test, train_img_pca, test_img_pca = pca_percentage(x_train_flat, x_test_flat, y_train, y_test, 0.9)
    # sequential_model_fit(128, 10, 20, 'accuracy', train_img_pca, y_train, test_img_pca, y_test, 99)