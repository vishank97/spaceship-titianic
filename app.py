import streamlit as st
import pandas as pd
import pickle 

#Loading up the Regression model we created
dbfile = open('xgb_model.pkl', 'rb')   
model = pickle.load(dbfile)

#Caching the model for faster loading
@st.cache
def predict(HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name):
    prediction = model.predict(pd.DataFrame([[HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name]], columns=['HomePlanet',
'CryoSleep',
'Cabin',
'Destination',
'Age',
'VIP',
'RoomService',
'FoodCourt',
'ShoppingMall',
'Spa',
'VRDeck',
'Name']))
    return prediction


st.title('Spaceship Titanic Predictor')
st.image("""https://www.thestreet.com/.image/ar_4:3%2Cc_fill%2Ccs_srgb%2Cq_auto:good%2Cw_1200/MTY4NjUwNDYyNTYzNDExNTkx/why-dominion-diamonds-second-trip-to-the-block-may-be-different.png""")
st.header('Enter the characteristics of the passenger:')



HomePlanet = st.number_input('HomePlanet:', min_value=0.0, max_value=10000.0, value=0.0)
CryoSleep = st.number_input('CryoSleep:', min_value=0.0, max_value=10000.0, value=0.0)
Cabin = st.number_input('Cabin:', min_value=0.0, max_value=10000.0, value=0.0)
Destination = st.number_input('Destination:', min_value=0.0, max_value=10000.0, value=0.0)
Age = st.number_input('Age:', min_value=0.0, max_value=10000.0, value=0.0)
VIP = st.number_input('VIP:', min_value=0.0, max_value=10000.0, value=0.0)
RoomService = st.number_input('RoomService:', min_value=0.0, max_value=10000.0, value=0.0)
FoodCourt = st.number_input('FoodCourt:', min_value=0.0, max_value=10000.0, value=0.0)
ShoppingMall = st.number_input('ShoppingMall:', min_value=0.0, max_value=10000.0, value=0.0)
Spa = st.number_input('Spa:', min_value=0.0, max_value=10000.0, value=0.0)
VRDeck = st.number_input('VRDeck:', min_value=0.0, max_value=10000.0, value=0.0)
Name = st.number_input('Name:', min_value=0.0, max_value=10000.0, value=0.0)

if st.button('Predict Survival'):
    survival = predict(HomePlanet,
CryoSleep,
Cabin,
Destination,
Age,
VIP,
RoomService,
FoodCourt,
ShoppingMall,
Spa,
VRDeck,
Name)
    st.success('The person is survived: '+str(survival))


