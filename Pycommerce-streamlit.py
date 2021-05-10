# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import numpy as np

df_events = pd.read_csv("C:/Users/lubke/Documents/Data projet Ecommerce/events.csv")
df_events.head()

user_value_counts = df_events['visitorid'].value_counts()
data = df_events[df_events['visitorid'].isin(user_value_counts[user_value_counts >= 10].index)]

#Selection des utilisateurs ayant réalisé un achat
customer_purchased = data[data.transactionid.notnull()].visitorid.unique()
    
purchased_items = []
buyer = []

    
# Création d'une liste qui contient leurs achats
for customer in customer_purchased:
    buyer.append(customer)
    purchased_items.append(list(data.loc[(data.visitorid == customer) & (data.transactionid.notnull())].itemid.values)) 
    
    
purchased_items_df = pd.DataFrame({"Item acheté":purchased_items})
buyer_df = pd.DataFrame({"visitorid":buyer, "Item acheté":purchased_items})
buyer_df.head()

buyer_df.set_index('visitorid', inplace=True)

 #STREAMLIT 1 Utilisateurs \ Achats
st.title('Système de recommandation')


option = st.selectbox(
    'Choisis un utilisateur ',
     buyer_df.index)

Id_acheteur	 = option

achat = buyer_df.loc[buyer_df.index == Id_acheteur, 'Item acheté'].iloc[0]
test_list = list(map(int, achat))

st.write("Voici la liste des achats réalisés par l’utilisateur sélectionné :", test_list)

#Fin Streamlit 1

#importation d'apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules 

#Transformation
te = TransactionEncoder()
te_ary = te.fit(purchased_items).transform(purchased_items)
df = pd.DataFrame(te_ary, columns=te.columns_)

#On défini le seuil minimum de prise en charge sur 0.001
frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

rules_ap = association_rules(frequent_itemsets, metric ="confidence", min_threshold = 0.2)

#creation d'un nouveau df dans le bon format (sans frozenset)
antecedents = rules_ap["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
consequents = rules_ap["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
confidence =  rules_ap["confidence"]
df_suggestion = pd.DataFrame({'antecedents':antecedents ,'consequents':consequents, "confidence" : confidence})


#STREAMLIT 2 Produits \ Recomandation

#option = st.number_input("Veuillez entrer un numéro de produit", value = 456056)

option2 = st.selectbox(
    'Choisis un produit',
    df_suggestion.antecedents)

produit	 = option2

reco = df_suggestion.loc[df_suggestion.antecedents == produit, 'consequents'].iloc[0]
confiance = df_suggestion.loc[df_suggestion.antecedents == produit, 'confidence'].iloc[0]

arrondi = confiance * 100
pourcentage = str(round(arrondi, 2))


st.write( pourcentage, "% des visiteurs ayant acheté cet article ont également acheté l’article", reco )

#Fin Streamlit 2



