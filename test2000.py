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

 
st.title('Systeme de recomendation')


option = st.selectbox(
    'Choisis un utilisateur?',
     buyer_df.index)

Id_acheteur	 = option

achat = buyer_df.loc[buyer_df.index == Id_acheteur, 'Item acheté'].iloc[0]
test_list = list(map(int, achat))

st.write("Voici la liste des achats réalisé par l'utilisateur sélectionné :", test_list)




