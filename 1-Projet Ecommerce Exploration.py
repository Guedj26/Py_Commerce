#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# * Import du premier fichier item properties - REMPLACER PAR BON CHEMIN 
# 

# In[2]:


df_properties1 = pd.read_csv("/Volumes/HD 2/DataScientest/PROJET/Jeu de données/donnees_brutes/item_properties_part1.csv")
df_properties1.head()


# * Import du second fichier item properties - REMPLACER PAR BON CHEMIN 

# In[3]:


df_properties2 = pd.read_csv("/Volumes/HD 2/DataScientest/PROJET/Jeu de données/donnees_brutes/item_properties_part2.csv")
df_properties2.head()


# * Concaténation des 2 fichiers item properties dans df_properties car fichiers séparés cause nombre de lignes trop important

# In[4]:


df_properties = pd.concat([df_properties1, df_properties2])
df_properties.head()


# * Affichage des infos de df_properties et recherche de valeurs nulles

# In[5]:


df_properties.info()
df_properties.isna().sum()


# _Nous donne 20275902 lignes et aucune valeur nulle_

# * Nombre valeurs uniques dans df_properties

# In[6]:


df_properties.nunique(axis=0)


# * Équilibre des valeurs sur df_properties timestamp (calcul pour compléter le fichier rapport d'exploration)

# In[7]:


print(df_properties['timestamp'].count())
print(df_properties['timestamp'].unique())
print(df_properties['timestamp'].value_counts())


# * Équilibre des valeurs sur df_properties itemid (calcul pour compléter le fichier rapport d'exploration)

# In[8]:


print(df_properties['itemid'].count())
print(df_properties['itemid'].unique())
print(df_properties['itemid'].value_counts())


# * Équilibre des valeurs sur df_properties property (calcul pour compléter le fichier rapport d'exploration)

# In[9]:


print(df_properties['property'].count())
print(df_properties['property'].unique())
print(df_properties['property'].value_counts())


# * Équilibre des valeurs sur df_properties value (calcul pour compléter le fichier rapport d'exploration)

# In[10]:


print(df_properties['value'].count())
print(df_properties['value'].unique())
print(df_properties['value'].value_counts())


# * Import du  fichier des catégories - REMPLACER PAR BON CHEMIN

# In[11]:


cat_tree = pd.read_csv("/Volumes/HD 2/DataScientest/PROJET/Jeu de données/donnees_brutes/category_tree.csv")
cat_tree.head()


# * Infos de cat_tree et recherche de valeurs nulles

# In[12]:


cat_tree.info()
cat_tree.isna().sum()


# _Nous donne 1669 rows ainsi que 25 valeurs nulles dans la colonne parentid_

# * Pourcentage de valeurs nulles trouvées dans cat_tree :

# In[13]:


na_parentid = (cat_tree.parentid.isna().sum() / cat_tree.shape[0]) * 100
print("parentid a", na_parentid, "% de na")


# * Nombre valeurs uniques dans cat_tree :

# In[14]:


cat_tree.nunique(axis=0)


# * Équilibre des valeurs sur cat_tree categoryid (calcul pour compléter le fichier rapport d'exploration)

# In[15]:


print(cat_tree['categoryid'].count())
print(cat_tree['categoryid'].unique())
print(cat_tree['categoryid'].value_counts())


# * Équilibre des valeurs sur cat_tree parentid (calcul pour compléter le fichier rapport d'exploration)

# In[16]:


print(cat_tree['parentid'].count())
print(cat_tree['parentid'].unique())
print(cat_tree['parentid'].value_counts())


# * Import du fichier des événements, jeu de données comportementales - REMPLACER PAR BON CHEMIN
# 
# 

# In[17]:


df_events = pd.read_csv("/Volumes/HD 2/DataScientest/PROJET/Jeu de données/donnees_brutes/events.csv")
df_events.head()


# * Infos de df_events et recherche de valeurs nulles

# In[18]:


df_events.info()
df_events.isna().sum()


# _Donne 2756101 rows et 2733644 valeurs nulels dans transaction id reflétant un nombre de transactions très bas_

# * Nombre valeurs uniques dans df_events :

# In[19]:


df_events.nunique(axis=0)


# _1407580 Utilisateurs unique,_
# _235061 Produits,_
# _17672 transactions_

# * Nombre de lignes dupliquées (que dans df_events, aucune dans les autres dataframes)

# In[20]:


df_events[df_events.duplicated()].shape


# _On constate 460 lignes dupliquées, on les supprime du dataframe df_events :_

# In[21]:


df_events.drop_duplicates(inplace=True)


# * Répartition du nombre de valeurs dans la colonne event

# In[22]:


df_events.event.value_counts()


# _2664218 produits consultés,_
# _68966 produits ajoutés au panier,_
# _22457 produits vendus_

# * Avec si peu de transactions on souhaite visualiser le pourcentage de valeurs nulles que cela représente :

# In[23]:


df_events.isna().mean()


# _On a 99% de valeurs nulles sur transactionid_

# * On souhaite regarder le nombre de valeurs non nulles quand l'événement n'est pas une transaction.

# In[24]:


pd.crosstab(df_events['event'], df_events['transactionid'].isna())


# _Le 0 indique que la variable transactionid est manquante uniquement lorsque le type d'événement n'est pas une transaction_

# * df_events est le dataframe qui va le plus nous intéresser pour des calculs d'événements
# * Conversion du timestamp en dates réelles dans df_events (changement de l'unix en date et creation d'une colonne jour de la semaine (les heures n'avaient pas de sens donc décalage de 5h pour match le -5 UTC americain) :

# In[25]:


import datetime
import calendar

df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit= "ms",) - datetime.timedelta(hours=5)
df_events['day_of_week'] = df_events.timestamp.dt.day_name()

df_events.head()


# * On remet les jours de la semaine dans l'ordre et on sépare les valeurs de la colonne event en 3 colonnes

# In[26]:


cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_events['day_of_week'] = pd.Categorical(df_events['day_of_week'], categories=cats, ordered=True)

df_events = df_events.join(pd.get_dummies(df_events.event, prefix = "events"))
df_events.head()


# * Durée data set df_events

# In[27]:


df_events.timestamp.max() - df_events.timestamp.min()


# _Soit une période de 4,5 mois_

# * Max et min de la variable timestamp convertie dans df_events

# In[28]:


df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
df_events['timestamp'].head()
maxi = df_events.max(axis=0)['timestamp']
mini = df_events.min(axis=0)['timestamp']
print('la date de debut du dataset est le', mini ,"\n",'et la date de fin du dataset est le', maxi)


# * Pourcentage de valeurs nulles dans df_events, colonne transaction :

# In[29]:


na_transaction = (df_events.transactionid.isna().sum() / df_events.shape[0]) * 100
print("transactionid a ", na_transaction, "% de na")


# * Équilibre des valeurs sur df_events timestamp (calcul pour compléter le fichier rapport d'exploration)

# In[30]:


print(df_events['timestamp'].count())
print(df_events['timestamp'].unique())
print(df_events['timestamp'].value_counts())


# * Équilibre des valeurs sur df_events visitorid  (calcul pour compléter le fichier rapport d'exploration)

# In[31]:


print(df_events['visitorid'].count())
print(df_events['visitorid'].unique())
print(df_events['visitorid'].value_counts())


# * On constate un déséquilibre, le visitorid 1150086 est venu 7757 fois, le visitorid 530559 est venu 4328 fois, suspicion de bots
# * Pour identifier s'il y a des bots on définie des bornes arbitrairement et on classe selon ce qu'on estime être des fréquences d'utilisateurs faibles à excessives. Au-dessus de 3 000, on classe ces visitorid dans botts

# In[32]:


df_events['User frequency'] = df_events['visitorid'].map(df_events['visitorid'].value_counts())
borne = [0, 10, 50, 3000, 7757]
label = ["Infrequent user","Recurrent user","Big user","Bots"]
df_events["User category"] =  pd.cut(df_events['User frequency'],borne, labels = label)
df_events.head()


# * On compte les nombre de valeurs de ces nouvelles catégories. Cela nous révèle un nombre anormal de fréquences élevées

# In[33]:


df_events['User category'].value_counts()


# * On fait la même manipulation sur les items pour mesurer la popularité de ces derniers 

# In[34]:


df_events['Product frequency'] = df_events['itemid'].map(df_events['itemid'].value_counts())
borne = [0,100, 1000, 1500, 2000]
label = ["Unpopular products","Popular products","Very popular products","Most popular products"]
df_events['Product category'] =  pd.cut(df_events['Product frequency'],borne, labels = label)
df_events.head()


# * Équilibre des valeurs sur df_events event  (calcul pour compléter le fichier rapport d'exploration)

# In[35]:


print(df_events['event'].count())
print(df_events['event'].unique())
print(df_events['event'].value_counts())


# * Équilibre des valeurs sur df_events itemid  (calcul pour compléter le fichier rapport d'exploration)

# In[36]:


print(df_events['itemid'].count())
print(df_events['itemid'].unique())
print(df_events['itemid'].value_counts())


# In[37]:


df_events.head()


# * Équilibre des valeurs sur df_events transactionid  (calcul pour compléter le fichier rapport d'exploration)

# In[38]:


print(df_events['transactionid'].count())
print(df_events['transactionid'].value_counts())


# • Comptage des valeurs uniques dans le dataframe

# In[39]:


a = len(df_events['itemid'].unique())
b = len(df_events['visitorid'].unique())
c = len(df_events['transactionid'].unique())
print('on observe', a,'produits uniques,', b,'visiteurs uniques et', c, 'transactions uniques dans le dataframe')


# * Observation de quelques "super-paniers" avec des paniers contenant de nombreux articles (ici test 28 articles, même ID de transaction)

# In[40]:


df_events.loc[df_events['transactionid'] == 765.0]


# * Produits les plus populaires (vue, panier, transaction confondues) sur df_events

# In[41]:


top_produits = df_events['itemid'].value_counts()
top_produits.sort_values(ascending=False)
top_produits.head(10)


# * Produits les moins populaires (vue, panier, transaction confondues) sur df_events

# In[42]:


top_produits.tail(10)


# * On souhaite réaliser l'étude statistique d'une variable cible (events_transaction) avec d'autres variables quantitatives afin d'analyser la corrélation inter-variables :

# In[43]:


from scipy.stats import pearsonr
pd.DataFrame(pearsonr(df_events['events_transaction'],df_events['User frequency']), index = ['pearson_coeff','p-value'], columns = ['result'])


# _Avec une p-value de 0 et un coefficient > 5%, on observe une forte corrélation entre les variables "events_transaction" et "User frequency", ce qui semble logique compte tenu des varaibles considérées._
# 
# * Si on réalise la même étude variable cible (events_transaction) et "Product frequency", on observe une corrélation beaucoup moins marquée car le coefficient est supérieur à 1 et la p-value s'approche de 5%. On peut tout de même considérer qu'il y a corrélation :

# In[44]:


pd.DataFrame(pearsonr(df_events['events_transaction'],df_events['Product frequency']), index = ['pearson_coeff','p-value'], columns = ['result'])


# Si l'on cherche à croiser la variable cible (events_transaction) et "events_view", on observe avec surprise un coefficient négatif. Cela nous indique que les variables, bien que corrélées par p-value évoluent en sens opposé :

# In[45]:


pd.DataFrame(pearsonr(df_events['events_transaction'],df_events['events_view']), index = ['pearson_coeff','p-value'], columns = ['result'])


# En croisant la variable cible (events_transaction) et "events_addtocart", on observe que les variables sont peu corrélées et évoluent en sens opposé, indiquant un taux d'abandon de panier très élevé et une rupture très marquée dans le parcours d'achat :

# In[46]:


pd.DataFrame(pearsonr(df_events['events_transaction'],df_events['events_addtocart']), index = ['pearson_coeff','p-value'], columns = ['result'])


# * L'utilisation de la fonction crosstab est utile pour identifier une corrélation entre deux variables quantitatives. Elle permet de croiser les différentes catégories de deux variables. On créer ici la table de contingence entre la variable cible events_transaction et User category :

# In[47]:


table = pd.crosstab(df_events['events_transaction'],df_events['User category'])
table


# * Puis on réalise un test de proportions entre l'effectif de la cellule et l'effectif total de la colonne, ici le χ2 sur la table de contingence déterminée précédemment (variable cible events_transaction et day of week)

# In[48]:


from scipy.stats import chi2_contingency

resultats_test = chi2_contingency(table)
statistique = resultats_test[0]
p_valeur = resultats_test[1]
degre_liberte = resultats_test[2]

print(statistique,p_valeur,degre_liberte)


# * Un dernier test V Cramer nous permet de voir la corrélation entre les deux variables :

# In[49]:


def V_Cramer (table,N):
    stat_chi2 = chi2_contingency(table)[0]
    k = table.shape[0]
    r = table.shape[1]
    phi = max(0,(stat_chi2/N)-((k-1)*(r-1)/(N-1)))
    k_corr = k - (np.square(k-1)/(N-1))
    r_corr = r - (np.square(r-1)/(N-1))
    return np.sqrt(phi/min(k_corr - 1,r_corr - 1))

V_Cramer(table,df_events.shape[0])


# * On observe que le V de Cramer donne un résultat relativement proche de 0, donc il existe une corrélation entre les variables events_transaction et day_of_week. Si on regarde la table de contingence on remarque que les jours de semaine il y a plus de ventes que le Week-end. Le jour avec le nombre de transactions le plus élevé est le mercredi. 
# 
# * On réalise les mêmes manipulations statistiques avec d'autres varaibles et la variable cible. Table de contingence entre les variable events_view et Product category :

# In[50]:


table = pd.crosstab(df_events['events_transaction'],df_events['Product category'])
table


# * Puis on teste le χ2 sur la table de contingence déterminée précédemment (variables events_transaction et Product category)
# 
# 

# In[51]:


resultats_test = chi2_contingency(table)
statistique = resultats_test[0]
p_valeur = resultats_test[1]
degre_liberte = resultats_test[2]

print(statistique,p_valeur,degre_liberte)


# * Enfin on réalise un test de V_Cramer par rapport au tableau de contingence (variables events_transaction et Product category) :

# In[52]:


def V_Cramer (table,N):
    stat_chi2 = chi2_contingency(table)[0]
    k = table.shape[0]
    r = table.shape[1]
    phi = max(0,(stat_chi2/N)-((k-1)*(r-1)/(N-1)))
    k_corr = k - (np.square(k-1)/(N-1))
    r_corr = r - (np.square(r-1)/(N-1))
    return np.sqrt(phi/min(k_corr - 1,r_corr - 1))

V_Cramer(table,df_events.shape[0])


# * On observe que le V de Cramer donne également un résultat relativement proche de 0, donc il existe une corrélation entre les variables events_transaction et Product category. En observant la table de contingence on remarque que les produits qui gènèrent le plus d'événements ne sont pas ceux qui se vendent le plus. Ce sont les produits de la catégorie 'Very popular products' qui ont le pourcentage d'événements de vente le plus important.
# 
# * On créer un nouveau dataframe qui va être utilisé pour le clustering et qui marque pour chaque visitorid le nombre de fois où il a un événement de type nombre de vies, nombre d'ajouts au panier et de transactions :
