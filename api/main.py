from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from api.titanic import infer_model


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Spaceship Titanic model inference!"}

class SpaceshipTitanic(BaseModel):
    # name: str
    # description: Union[str, None] = None
    # price: float
    # tax: Union[float, None] = None

    HomePlanet: str
    CryoSleep: int
    Cabin: str
    Destination: str
    Age: float
    VIP: int
    RoomService: float
    FoodCourt: float
    ShoppingMall: float
    Spa: float
    VRDeck: float

@app.post("/infer")
async def infer(st: SpaceshipTitanic):
    return infer_model(
       st.HomePlanet,
       st.CryoSleep,
       st.Cabin,
       st.Destination,
       st.Age,
       st.VIP,
       st.RoomService,
       st.FoodCourt,
       st.ShoppingMall,
       st.Spa,
       st.VRDeck
    )