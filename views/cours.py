import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
import base64

def write():
    st.title('La PCA, comprendre et appliquer')

    st.header('1) Introduction')
    st.write("L'analyse en composantes principales (ACP) est essentielle pour la science des données, l'apprentissage automatique, la visualisation des données, les statistiques et d'autres domaines quantitatifs. L’ACP est cruciale et stratégique pour un data analyst. Il est donc très important de bien la comprendre. Elle nécessite un minimum de pratique pour analyser les données correctement. Si la compréhension est approximative, elle mène alors très facilement à des analyses erronées, approximatives")

    st.header('2) Contexte')
    st.write("Lors de la mise en œuvre d'algorithmes d'apprentissage automatique, l'inclusion d'un plus grand nombre de features peut entraîner une aggravation des problèmes de performance. L'augmentation du nombre de features n'améliore pas toujours la précision de la classification, ce qui est également connu comme : le fléau de la dimensionnalité.")
    image_1 = Image.open('img/dimensionality_vs_performance.png')
    st.image(image_1, caption='Sunrise by the mountains')

    st.header('3) Problématique')
    st.write("Lorsqu’on étudie simultanément un nombre important de variables quantitatives (ne serait-ce que 4 !), comment en faire un graphique global ? La difficulté vient de ce que les individus étudiés ne sont plus représentés dans un plan, espace de dimension 2, mais dans un espace de dimension plus importante (par exemple  4).  L’objectif  de  l’Analyse  en  Composantes  Principales  (ACP)  est de revenir à un espace de dimension réduite (par exemple 2) en déformant le moins possible la réalité. Il s’agit donc d’obtenir le résumé le plus pertinent possible des données initiales.")

    st.header('4) Principe')
    st.write("Le principe de l'ACP est d'appliquer une réduction de la dimensionnalité pour améliorer la précision de la classification en sélectionnant l'ensemble optimal de caractéristiques de plus faible dimensionnalité..")
    image_2= Image.open('img/projection.png')
    st.image(image_2, caption='Sunrise by the mountains')

    st.header('5) Fonctionnement')
    st.write("C’est la matrice des variances-covariances (ou celle des corrélations) qui va permettre de réaliser ce résumé pertinent, parce qu’on analyse essentiellement : la dispersion des données considérées. De cette matrice, on va extraire, par un procédé mathématique adéquat, les facteurs que l’on recherche, en petit nombre. Ils vont permettre de réaliser les graphiques désirés dans cet espace de petite dimension (le nombre de facteurs retenus), en déformant le moins possible la configuration globale des individus selon l’ensemble des variables initiales (ainsi remplacées par les facteurs). D’un point de vue plus “mathématique”, l’ACP correspond à l’approximation d’une matrice (n,p) par une matrice de même dimensions mais de rang q < p ; q étant souvent de petite valeur (2, 3) pour la construction de graphiques facilement compréhensibles.")

    st.header('6) Points de vigilance')
    st.subheader('6.1 Les données doivent être standardisées')
    st.write("L'objectif est de pouvoir comparer les variables. Généralement les variables sont mises à l'échelle pour avoir un écart-type de un et une moyenne de zéro.")
    st.subheader("6.2 L'analyse en composantes principales est conçue pour les variables continues.")
    st.write("Bien que l'on puisse utiliser l'ACP sur des données binaires (par exemple en encodant avec one-hot), cela ne signifie pas que ce soit une bonne manière de faire, ou que cela donnera des résultats pertinents. L'ACP essaye de minimiser la variance (= écarts quadratiques). Le concept d'écart au carré s'effondre lorsque vous avez des variables binaires. Donc oui, vous pouvez utiliser l'ACP. Et oui, vous obtenez un résultat. Il s'agit même d'un résultat des moindres carrés - ce n'est pas comme si l'ACP était défaillante sur de telles données. Cela fonctionne, mais c'est beaucoup moins significatif que vous ne le souhaiteriez, et supposément moins significatif que, par exemple, l'extraction de motifs fréquents.")
    st.subheader('6.3 Représentation de la variance')
    st.write("Nous devons prendre les vecteurs propres qui représentent le mieux notre dataset. Ce sont les vecteurs qui ont les valeurs propres les plus élevées. Selon Machine Learnia il est habituel de prendre les vecteurs capturant au moins 90 à 95% de la variance.")
    image_3= Image.open('img/variance.png')
    st.image(image_3, caption='Sunrise by the mountains')

    st.header("7) Comprendre l'ACP")
    st.write("- Aperçu du processus :")
    st.write("1. La technique de l'ACP analyse l'ensemble des données et trouve ensuite les points dont la variance est maximale.")
    st.write(" 2. Elle crée de nouvelles variables de sorte qu'il existe une relation linéaire entre les nouvelles variables et les variables d'origine, de sorte que la variance soit maximisée.")
    st.write("3. Une matrice de covariance est ensuite créée pour les caractéristiques afin de comprendre leur multi-collinéarité.")
    st.write("4. Une fois la matrice de variance-covariance calculée, l'ACP utilise les informations recueillies pour réduire les dimensions. Elle calcule des axes orthogonaux à partir des axes originaux des caractéristiques. Ce sont les axes des directions ayant une variance maximale. Ces vecteurs représentent les directions de variance maximale qui sont connues comme 'les composantes principales'. On crée ensuite les valeurs propres qui définissent la magnitude des composantes principales. Les valeurs propres sont les composantes de l'ACP.")
    st.write("Par conséquent, pour N dimensions, il y aura une matrice de variance-covariance NxN et par conséquent, nous aurons un vecteur propre de N valeurs et une matrice de N valeurs propres.")

    st.header("8) Ce que sont les vecteurs propres et les valeurs propres")
    st.subheader("8.1  Vecteur propre")
    st.write("Si nous multiplions une matrice par un vecteur, nous obtenons un nouveau vecteur. La multiplication d'une matrice par un vecteur est connue sous le nom de matrices de transformation. Nous pouvons transformer et changer les matrices en nouveaux vecteurs en multipliant une matrice par un vecteur. La multiplication de la matrice par un vecteur permet de calculer un nouveau vecteur. C'est le vecteur transformé.")
    st.write("- Le nouveau vecteur peut être considéré sous deux formes :")
    st.write("1 Parfois, le nouveau vecteur transformé est juste une forme mise à l'échelle du vecteur original. Cela signifie que le nouveau vecteur peut être recalculé en multipliant simplement un scalaire (nombre) au vecteur original.")
    st.write("2. D'autres fois, le vecteur transformé n'a pas de relation scalaire directe avec le vecteur original que nous avons utilisé pour le multiplier à la matrice")
    st.write("Si le nouveau vecteur transformé est juste une forme mise à l'échelle du vecteur original, alors le vecteur original est connu pour être un vecteur propre de la matrice originale. Les vecteurs qui présentent cette caractéristique sont des vecteurs spéciaux et sont appelés 'vecteurs propres'. Les vecteurs propres peuvent être utilisés pour représenter une matrice de grande dimension.")

    file_ = open("img/vecteurspropres.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,)

    st.subheader("8.2  Valeur propre")
    st.write("Le scalaire qui est utilisé pour transformer (étirer) un vecteur propre.")

    st.header("Les composantes principales ")
    st.write("Les composantes principales sont la clé de la PCA. *Lorsque les données sont projetées dans une dimension inférieure (supposons trois dimensions) à partir d'un espace supérieur, les trois dimensions ne sont rien d'autre que les trois composantes principales qui capturent (ou conservent) la majeure partie de la variance (informations) de vos données.")
    st.write("Les composantes principales ont à la fois une direction et une amplitude.")
    st.write("- La direction représente sur quels axes principaux les données sont principalement réparties ou ont le plus de variance.")
    st.write("- L'amplitude signifie la quantité de variance que la composante principale capture des données lorsqu'elles sont projetées sur cet axe.")
    st.write("Les composantes principales sont une ligne droite et la première composante principale contient le plus de variance des données. Chaque composante principale suivante est orthogonale à la dernière et a une variance moindre. De cette façon, étant donné un ensemble de x variables corrélées sur y échantillons, vous obtenez un ensemble de u composantes principales non corrélées sur les mêmes y échantillons.")
    st.write("La raison pour laquelle vous obtenez des composantes principales non corrélés à partir des entités d'origine est que les entitées corrélées contribuent à la même composante principale.")
    st.write("Chaque composante principale représente un ensemble différent de caractéristiques corrélées avec différentes quantités de variation.")


    image_4= Image.open('img/dependencies.png')
    st.image(image_4, caption='Sunrise by the mountains')
    st.write("Chaque composante principale représente un pourcentage de la variation totale capturée à partir des données.")

    st.header("10) Utilisations ")
    st.write("- Réduire le nombre de dimensions du dataset.")
    st.write("- Trouver des patterns dans les datasets de grande dimension.")
    st.write("- Pour visualiser les données de grande dimension.")
    st.write("- Pour ignorer le bruit.")
    st.write("- Améliorer la classification.")
    st.write("- Obtenir une description compacte.")
    st.write("- Capter autant que possible la variance originale des données.")

    st.header("11) ")
    st.write("- Visualisation de données.")
    st.write("- Compression de données.")
    st.write("- Réduction du bruit.")
    st.write("- Classification des données.")
    st.write("- Compression d'images.")
    st.write("- Reconnaissance des visages.")
    # Applications de la ACP :





