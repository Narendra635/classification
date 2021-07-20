# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
#from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd
import json
import ast
import re
from copy import deepcopy
from fastapi.middleware.cors import CORSMiddleware


# 2. Create the app object
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


loaded_model = pickle.load(open( 'ClassificationModel.pkl', 'rb'))
cv = pickle.load(open( 'transform.pkl', 'rb'))
la = pickle.load(open( 'labeltransform.pkl', 'rb'))


#pickle_in = open("classifier.pkl","rb")
#classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
#@app.get('/{name}')
#def get_name(name: str):
 #   return {'Welcome To AI World': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_narrative(narrative: str):
    nararray=[narrative]
    xt=cv.transform(nararray)
    loaded_model.predict(xt)
    output=la.inverse_transform(loaded_model.predict(xt))
    output = [item for items in output for item in items]
    response_data = []
    for i in output:
        response_data.append(json.loads(re.sub(r",\s*(\w+)", r", '\1'", re.sub(r"\s*\{\s*(\w+)", r"{'\1'", i)).replace("'", '"')))

    print(response_data)
    res = dict()
    for dic in response_data:
        for key, val in dic.items():
            if key in res:

                res[key].update(val)
            else:
                res[key] = val
    response = []
    response.append(res)
    print('Final->>> ',response)
    if len(output)==0 : output=[{'generalNeglect': {'inadequateSupervision': True}},{'riskOfHarm': {'previousfatality': True}}]


    return {
        'data':json.dumps(response)
    }


@app.get('/{narrative}')
def predict_narrative(narrative: str):
    nararray=[narrative]
    xt=cv.transform(nararray)
    loaded_model.predict(xt)
    output=la.inverse_transform(loaded_model.predict(xt))
    output = [item for items in output for item in items]
    response_data = []
    for i in output:
        response_data.append(json.loads(re.sub(r",\s*(\w+)", r", '\1'", re.sub(r"\s*\{\s*(\w+)", r"{'\1'", i)).replace("'", '"')))

    print(response_data)
    res = dict()
    for dic in response_data:
        for key, val in dic.items():
            if key in res:

                res[key].update(val)
            else:
                res[key] = val
    response = []
    response.append(res)
    print('Final->>> ',response)
    if len(output)==0 : output=[{'generalNeglect': {'inadequateSupervision': True}},{'riskOfHarm': {'previousfatality': True}}]


    return {
        'data':response
    }    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
##{"mode":"full","isActive":false}
