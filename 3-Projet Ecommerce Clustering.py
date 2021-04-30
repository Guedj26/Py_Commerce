#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# * Import du fichier des événements, jeu de données comportementales - REMPLACER PAR BON CHEMIN

# In[3]:


df_events = pd.read_csv("/Volumes/HD 2/DataScientest/PROJET/Jeu de données/donnees_brutes/events.csv")
df_events.head()


# * df_events est le dataframe qui va le plus nous intéresser pour des calculs d'événements
# * Conversion du timestamp en dates réelles dans df_events (changement de l'unix en date et creation d'une colonne jour de la semaine (les heures n'avaient pas de sens donc décalage de 5h pour match le -5 UTC americain)
# * On remet les jours de la semaine dans l'ordre et on sépare les valeurs de la colonne event en 3 colonnes

# In[4]:


import datetime
import calendar

df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit= "ms",) - datetime.timedelta(hours=5)
df_events['day_of_week'] = df_events.timestamp.dt.day_name()

cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_events['day_of_week'] = pd.Categorical(df_events['day_of_week'], categories=cats, ordered=True)

df_events = df_events.join(pd.get_dummies(df_events.event, prefix = "events"))

df_events.head()


# * On créer un nouveau dataframe qui regroupe les actions réalisées par chaque visiteur :

# In[5]:


df_id = df_events[df_events['events_view'] == 1]
df_id = df_id['visitorid'].value_counts()
df_id = df_id.to_frame()
df_id.columns = ['Nombre de vues']
df_id = df_id.rename_axis('visitorid').reset_index() 

cart = df_events[df_events['events_addtocart'] == 1]
cart = cart['visitorid'].value_counts()
cart = cart.to_frame()
cart.columns = ['Nombre d\'ajout au panier']
cart = cart.rename_axis('visitorid').reset_index() 
df_id = df_id.merge(right = cart, on = 'visitorid', how = 'outer' )

transac = df_events[df_events['events_transaction'] == 1]
transac = transac['visitorid'].value_counts()
transac = transac.to_frame()
transac.columns = ['Nombre de transactions']
transac = transac.rename_axis('visitorid').reset_index()
df_id = df_id.merge(right = transac, on = 'visitorid', how = 'outer' )
df_id.head()


# * Enfin on place les utilisateurs en index et on remplace les valeurs manquante par 0.

# In[6]:


df_id = df_id.set_index('visitorid')
df_id = df_id.fillna(0)


# * Essai de clustering hiérarchique :

# In[7]:


#ls_features = list(df_id.keys())

# Initialisation du classificateur CAH pour 4 clusters
#cluster = AgglomerativeClustering(n_clusters = 4)

# Apprentissage des données 
#cluster.fit(df_id[ls_features])

# Calcul des labels du data set
#labels = cluster.labels_


# In[8]:


#Vis
#plt.scatter(x = labels, y = df_id.index)


# _Le clustering hiérarchique n'est pas adapté a ce type de données. La base de donnée est trop grande et le kernel n'arrive pas a s'éxécuter._

# * Clustering avec K means où on cherche à visualiser les données

# In[9]:


plt.scatter(df_id['Nombre de vues'],df_id['Nombre de transactions'])
plt.xlabel('Nombre de vues')
plt.ylabel('Nombre de transactions')
plt.title('Visualisation des utilisateurs', bbox={'facecolor':'0.8', 'pad':5})
plt.show()


# _On observe que la majorité des utilisateurs ont consulté et acheté des produits de manière raisonable, cependant certains d'entre eux ont réalisé plus d'une centaine d'achats en 4 mois et ont consulté le site de manière trop fréquente pour un utilisateur lambda._
# 
# * L'objectif du clustering va être de repérer ces utilisateurs anormaux avec un diagramme en boîte (boxplot) de tous les variables explicatives

# In[10]:


liste = [df_id["Nombre de vues"],df_id["Nombre d'ajout au panier"], df_id["Nombre de transactions"] ]
plt.figure()
plt.title('Diagramme en boîte des actions des utilisateurs', bbox={'facecolor':'0.8', 'pad':5})
plt.boxplot(liste, labels = ['Vues', 'Ajout au panier', "Transactions"])
plt.show()


# _On observe égalememt sur ce diagramme en boîte des utilisateurs ayant des statistiques de visualisation ou d'achat qui sortent de l'ordinaire._

# * On réalise la méthode du coude afin de déterminer le juste nombre de clusters nécessaire :

# In[11]:


# Liste des nombres de clusters
range_n_clusters = [2, 3, 4, 5, 6]  

# Initialisation de la liste de distorsions
distorsions = []

# Calcul des distorsions pour les différents modèles
for n_clusters in range_n_clusters:
    
    # Initialisation d'un cluster ayant un pour nombre de clusters n_clusters
    cluster = KMeans(n_clusters = n_clusters)
    
    # Apprentissage des données suivant le cluster construit ci-dessus
    cluster.fit(df_id)
    
    # Ajout de la nouvelle distorsion à la liste des données
    distorsions.append(sum(np.min(cdist(df_id, cluster.cluster_centers_, 'euclidean'), axis=1)) / np.size(df_id, axis = 0))


# * Puis on visualise les distorsions en fonction du nombre de clusters

# In[12]:


plt.plot(range_n_clusters, distorsions, 'gx-')
plt.xlabel('Nombre de Clusters K')
plt.ylabel('Distorsion (WSS/TSS)')
plt.title('Méthode du coude affichant le nombre de clusters optimal', bbox={'facecolor':'0.8', 'pad':5})
plt.show()


# * Mise en place de l'algorithme des K-means avec 4 clusters (choisis grâce a la méthode du coude vue précédemment)

# In[13]:


# k-means
kmeans = KMeans(n_clusters=4, random_state=0)
df_id["cluster"] = kmeans.fit_predict(df_id[['Nombre de vues', 'Nombre de transactions']])

# get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]

## add to df
df_id['cen_x'] = df_id.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2],3:cen_x[3]})
df_id['cen_y'] = df_id.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2],3:cen_y[3]})

# define and map colors
colors = ['#bf20df', '#2095DF',"#AE6042", '#DF2020']
df_id['c'] = df_id.cluster.map({0:colors[0], 1:colors[1], 2:colors[2],3:colors[3]})


# * Visualisation du k-mean

# In[14]:


#Visualisation
from matplotlib.lines import Line2D
fig, ax = plt.subplots(1, figsize=(10,10))

# plot des data
plt.scatter(df_id['Nombre de vues'], df_id['Nombre de transactions'], c= df_id.c, alpha = 0.6, s=10)

# Legende
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
               markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]

# plot de la legende
plt.legend(handles=legend_elements, loc='upper left')

# titre et labels
plt.title("Visualisation des utilisateurs", bbox={'facecolor':'0.8', 'pad':5})
plt.xlabel('Nombre de vues')
plt.ylabel('Nombre de transactions')


# Grâce au clustering on observe 4 clusters distincts qui nous permettent d'isoler les utilisateurs ayant un comportement anormal.
# On peut émettre alors plusieurs hypothèses sur les utilisateurs présents dans le 3ème et 4ème clusters :
# - Il y a un bug et plusieurs clients utilisent le site avec le même id uilisateur.
# - Certains utilisateurs sont des clients b2b et font de très grosses commandes sur le site.
# - Certains utilisateurs sont en réalité des bots.

# **AJOUT DU 16/04 :**
# 
# 
# Cette partie a pour objectif de créer des dataframes qui montrent les produits achetés, ajoutés au panier ou vu par les utilisateurs groupés par cluster.
# Pour commencer on crée un dataframe qui reprend certaines données de df_id : le cluster de chaque utilisateur avec tous les évènements qui le concernent.

# In[15]:


df_id_cluster = df_id[['cluster','Nombre de transactions','Nombre d\'ajout au panier','Nombre de vues']]
df_id_cluster


# * On cherche à identifier le nombre de visiteurs qui correspondent à chaque cluster :

# _On observe une grande disproportion de visiteurs entre les clusters. Les clusters 1, 2 et 3 présentent des utilisateurs aux comprtements extrêmes et qu'il faudra traiter séparemment (traitement business, recherche d'un bug technique…étude à approfondir)._

# In[16]:


vuesdf = df_id_cluster.index[df_id['Nombre de vues'] == 0].nunique()
print(vuesdf, ' visiteurs, tous clusters confondus, sont venus sur le site sans regarder un seul item')

panierdf = df_id_cluster.index[df_id['Nombre d\'ajout au panier'] == 0].nunique()
print(panierdf, ' visiteurs, tous clusters confondus, sont venus sur le site sans ajouter au panier un seul item')

transacdf = df_id_cluster.index[df_id['Nombre de transactions'] == 0].nunique()
print(transacdf, ' visiteurs, tous clusters confondus, sont venus sur le site sans acheter un seul item')


# * Il est intéressant de visualiser par cluster le nombre de vues, d'ajouts au panier et de transactions :

# In[17]:


df_nb_views_cluster = df_id_cluster.groupby('cluster').sum()
df_nb_views_cluster.head()


# _Au regard du nombre de vues conséquent sur le cluster 0, il sera intéressant de le séparer en sous-clusters pour étudier la population de visiteurs plus finement. Ce qu'on fera ultérieurement._
# 
# * on créer un nouveau dataframe qui permet de lier les informations que l'on a sur chaque visiteur selon son cluster d'appartenance mais en y ajoutant l'itemid. Cela nous permet de voir quels produits ont été visités par visiteur.

# In[18]:


df_linked = df_events.merge(right  = df_id_cluster, on = 'visitorid', how = 'inner')
df_linked.head()


# * Le dataframe suivant représente le nombre total d'événements en fonction des clusters et de chaque produit :

# In[19]:


df_table = pd.crosstab(df_linked["itemid"],df_linked["cluster"])
sum_column = df_table[1] + df_table[2] + df_table[3] + df_table[0]
df_table['Total'] = sum_column
df_table = df_table.sort_values(by = 'Total', ascending = False)
df_table


# * Les dataframes suivants suivent les mêmes principes mais cette fois-ci en fonction du type d'événement : vue, ajout au panier et transaction :

# In[20]:


df_view = df_linked[df_linked['event'] != 'transactions']
df_view = df_view[df_view['event'] != 'addtocart']

table_view = pd.crosstab(df_view["itemid"],df_view["cluster"])
sum_column = table_view[1] + table_view[2] + table_view[3] + table_view[0]
table_view['Total'] = sum_column
table_view = table_view.sort_values(by = 'Total', ascending = False)
table_view


# In[21]:


df_addtocart = df_linked[df_linked['event'] != 'view']
df_addtocart = df_addtocart[df_addtocart['event'] != 'transactions']

table_addtocart = pd.crosstab(df_addtocart["itemid"],df_addtocart["cluster"])
sum_column = table_addtocart[1] + table_addtocart[2] + table_addtocart[3] + table_addtocart[0]
table_addtocart['Total'] = sum_column
table_addtocart = table_addtocart.sort_values(by = 'Total', ascending = False)
table_addtocart


# In[22]:


df_transaction = df_linked[df_linked['event'] != 'view']
df_transaction = df_transaction[df_transaction['event'] != 'addtocart']

table_transaction = pd.crosstab(df_transaction ["itemid"],df_transaction ["cluster"])
sum_column = table_transaction[1] + table_transaction[2] + table_transaction[3] + table_transaction[0]

table_transaction['Total'] = sum_column
table_transaction = table_transaction.sort_values(by = 'Total', ascending = False)
table_transaction


# _Afin d'augmenter la précision de nos analyses, nous allons nous concentrer sur le cluster 0 qui concentre 99.98% des utilisateurs._
# 
# * Pour cela nous allons faire une étude spécifiquement sur les utilisateurs du cluster 0. Création d'un nouveau dataframe contenant les utilisateurs du cluster 0 :

# In[23]:


df_new = df_id[df_id.cluster == 0]
df_new = df_new.drop(['cluster', 'cen_x', "cen_y", "c"], axis=1)
df_new.head()


# * On réalise à nouveau la méthode du coude afin de déterminer le juste nombre de clusters nécessaires :

# In[24]:


# Liste des nombre de clusters
range_n_clusters = [2, 3, 4, 5, 6]  

# Initialisation de la liste de dissortions
distorsions = []

# Calcul des distorsions pour les différents modèles
for n_clusters in range_n_clusters:
    
    # Initialisation d'un cluster ayant un pour nombre de clusters n_clusters
    cluster = KMeans(n_clusters = n_clusters)
    
    # Apprentissage des données suivant le cluster construit ci-dessus
    cluster.fit(df_new)
    
    # Ajout de la nouvelle distorsion à la liste des données
    distorsions.append(sum(np.min(cdist(df_new, cluster.cluster_centers_, 'euclidean'), axis=1)) / np.size(df_id, axis = 0))


# * Puis on visualise les distorsions en fonction du nombre de clusters :

# In[25]:


plt.plot(range_n_clusters, distorsions, 'gx-')
plt.xlabel('Nombre de Clusters K')
plt.ylabel('Distorsion (WSS/TSS)')
plt.title('Méthode du coude affichant le nombre de clusters optimal', bbox={'facecolor':'0.8', 'pad':5})
plt.show()


# * Mise en place de l'algorithme des K-means avec 4 clusters (choisis grâce a la méthode du coude vue précédemment)

# In[26]:


# k-means
kmeans = KMeans(n_clusters=4, random_state=0)
df_new["cluster"] = kmeans.fit_predict(df_new[['Nombre de vues', 'Nombre de transactions']])

# get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]

## add to df
df_new['cen_x'] = df_id.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2],3:cen_x[3]})
df_new['cen_y'] = df_id.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2],3:cen_y[3]})

# define and map colors
colors = ['#bf20df', '#2095DF',"#AE6042", '#DF2020']
df_new['c'] = df_new.cluster.map({0:colors[0], 1:colors[1], 2:colors[2],3:colors[3]})


# * Visualisation des nouveaux clusters créés a partir du cluster 0 :

# In[27]:


#Visualisation
from matplotlib.lines import Line2D
fig, ax = plt.subplots(1, figsize=(10,10))

# plot des data
plt.scatter(df_new['Nombre de vues'], df_new['Nombre de transactions'], c= df_new.c, alpha = 0.6, s=10)

# Legende
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
               markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]

# plot de la legende
plt.legend(handles=legend_elements, loc='upper left')

# titre et labels
plt.title("Visualisation des utilisateurs", bbox={'facecolor':'0.8', 'pad':5})
plt.xlabel('Nombre de vues')
plt.ylabel('Nombre de transactions')


# In[28]:


df_new.cluster.value_counts()


# _On observe une distribution beaucoup plus homogène entre les clusters.
# Cette distribution va nous permetre d'établir une segmentation plus précise, et d'augmenter la précision de notre algorithme de machine learning._

# **Par ces constats, visualisations et travaux de clustering réalisés, on souhaite prédire ce qu'un visiteur sera susceptible d'acheter, selon son ou ses précédents achats, afin de lui proposer des articles en lien avec ses intérêts et maximiser le taux de transformation : moteur de recommandation.**

# **Cédric, AJOUT 20/04**
# 
# • Dans une nouvelle variable cluster_0_classe, découpe des transactions de df_new en 4 classes distinctes avec pour labels 0, 1, 2, 3 selon les 3 quantiles de la nouvelle variable crée.

# In[29]:


# Firstly let's create an array that lists visitors who made a purchase
customer_purchased = df_events[df_events.transactionid.notnull()].visitorid.unique()
    
purchased_items = []
buyer = []
    
# Create another list that contains all their purchases 
for customer in customer_purchased:

    #Generate a Pandas series type object containing all the visitor's purchases and put them in the list
    buyer.append(customer)
    purchased_items.append(list(df_events.loc[(df_events.visitorid == customer) & (df_events.transactionid.notnull())].itemid.values))


# In[30]:


purchased_items_df = pd.DataFrame({"Purchased_items":purchased_items})
purchased_items_df.head()


# In[31]:


data = pd.DataFrame({"Buyer":buyer, "Purchased_items":purchased_items})
data.head()
data.info()


# In[ ]:





# Quelles ont nos variables explicatives ? TEST DA121
# 
# • Stockez dans une variable data, les données explicatives uniquement.
# 
# • Séparez les données en un ensemble d'apprentissage et un ensemble de test (20%), avec data comme données explicatives et charges_classes comme variable cible.
# 
# • Centrer et réduire les variables explicatives des deux échantillons de manière adéquate.
# 
# 
# 
# Dans la suite nous allons comparer plusieurs méthodes d'apprentissage. Pour chacune d'elles, il conviendra d'explorer le périmètre des hyperparamètres suivant :
# 
# K-plus proches voisins. Hyperparamètre à régler :
# 
# 'n_neighbors' : 2 à 50.
# SVM. Hyperparamètres à régler :
# 
# kernel : 'rbf', 'linear'.
# C : 0.1 ; 1 ; 10 ; 50 .
# RandomForest. Hyperparamètres à régler :
# 
# 'max_features': "sqrt", "log2", None
# 'min_samples_split': Nombres pairs allant de 2 à 30.
# Pour chaque algorithme mentionné ci-dessus:
# 
# Sélectionnez les hyperparamètres sur l'échantillon d’apprentissage par validation croisée
# Affichez les hyperparamètres retenus
# Appliquez le modèle à l'ensemble de test, affichez la matrice de confusion et le score du modèle sur ce dernier
# 
# Quel modèle fournit la meilleure précision ?
