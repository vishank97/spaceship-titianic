import pickle
import json
import pandas as pd

with open('api/xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)
f = open('api/encoding.json')
encoded_values = json.load(f)

def infer_model(HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck):
    # map the encoded values here
    to_predict = [encoded_values[HomePlanet],
            CryoSleep,
            encoded_values[Cabin],
            encoded_values[Destination],
            Age,
            VIP,
            RoomService,
            FoodCourt,
            ShoppingMall,
            Spa,
            VRDeck]

    test = pd.DataFrame(data=[to_predict],columns=['HomePlanet',
        'CryoSleep',
        'Cabin',
        'Destination',
        'Age',
        'VIP',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck'
    ])
    prediction = model.predict(test)
    return {'survived':int(prediction[0])}