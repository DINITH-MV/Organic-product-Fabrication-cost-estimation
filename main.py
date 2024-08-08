import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from openai import OpenAI
import re
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Set OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

# Function to classify image using OpenAI API
def classify_image(img_url):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me whether is it [bottle || container || bag || cup || plate || utensil || tub || wrap || tray || bucket]? Please provide a single-word description of the product type."},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300
    )

    # Ensure response is parsed as JSON
    result = response.model_dump_json()
    result = json.loads(result)
    
    # Extracting the product type from the response
    product_type = result['choices'][0]['message']['content'].strip().lower()  # Adjust based on actual response structure
    product_type = re.sub(r'[^\w\s]', '', product_type)  # Remove punctuation
    return product_type

# Updated templates for generating data based on product type and material type
product_templates = {
    'bottle': {
        'weight': [0.1, 0.12],  # kg
        'production_time': [0.5, 0.6],  # hours
        'labor_cost': [5.0, 6.0],  # USD
        'overhead_cost': [0.1, 0.15],  # USD
        'total_cost': [10.0, 15.0]  # USD (plastic vs. organic)
    },
    'container': {
        'weight': [0.2, 0.25],  # kg
        'production_time': [0.7, 0.8],  # hours
        'labor_cost': [7.0, 8.0],  # USD
        'overhead_cost': [0.2, 0.25],  # USD
        'total_cost': [12.0, 18.0]  # USD (plastic vs. organic)
    },
    'bag': {
        'weight': [0.05, 0.06],  # kg
        'production_time': [0.3, 0.35],  # hours
        'labor_cost': [2.0, 2.5],  # USD
        'overhead_cost': [0.05, 0.07],  # USD
        'total_cost': [4.0, 6.0]  # USD (plastic vs. organic)
    },
    'cup': {
        'weight': [0.15, 0.18],  # kg
        'production_time': [0.6, 0.7],  # hours
        'labor_cost': [6.0, 7.0],  # USD
        'overhead_cost': [0.15, 0.2],  # USD
        'total_cost': [12.0, 17.0]  # USD (plastic vs. organic)
    },
    'plate': {
        'weight': [0.1, 0.12],  # kg
        'production_time': [0.5, 0.6],  # hours
        'labor_cost': [5.0, 6.0],  # USD
        'overhead_cost': [0.1, 0.15],  # USD
        'total_cost': [10.0, 15.0]  # USD (plastic vs. organic)
    },
    'utensil': {
        'weight': [0.05, 0.06],  # kg
        'production_time': [0.4, 0.5],  # hours
        'labor_cost': [3.0, 3.5],  # USD
        'overhead_cost': [0.07, 0.1],  # USD
        'total_cost': [6.0, 8.0]  # USD (plastic vs. organic)
    },
    'tub': {
        'weight': [0.3, 0.35],  # kg
        'production_time': [0.8, 0.9],  # hours
        'labor_cost': [8.0, 9.0],  # USD
        'overhead_cost': [0.3, 0.35],  # USD
        'total_cost': [15.0, 22.0]  # USD (plastic vs. organic)
    },
    'wrap': {
        'weight': [0.02, 0.025],  # kg
        'production_time': [0.2, 0.25],  # hours
        'labor_cost': [1.0, 1.2],  # USD
        'overhead_cost': [0.02, 0.03],  # USD
        'total_cost': [2.0, 3.0]  # USD (plastic vs. organic)
    },
    'tray': {
        'weight': [0.25, 0.3],  # kg
        'production_time': [0.7, 0.8],  # hours
        'labor_cost': [7.0, 8.0],  # USD
        'overhead_cost': [0.2, 0.25],  # USD
        'total_cost': [12.0, 18.0]  # USD (plastic vs. organic)
    },
    'bucket': {
        'weight': [0.5, 0.6],  # kg
        'production_time': [1.0, 1.2],  # hours
        'labor_cost': [20.0, 25.0],  # USD
        'overhead_cost': [0.2, 0.3],  # USD
        'total_cost': [50.0, 70.0]  # USD (plastic vs. organic)
    }
}

# Generate DataFrame based on detected product type
def generate_data(product_class):
    if product_class in product_templates:
        template = product_templates[product_class]
        data = {
            'product_type': [product_class] * len(template['weight']),
            'material_type': ['plastic'] * len(template['weight']),
            'weight': template['weight'],
            'production_time': template['production_time'],
            'labor_cost': template['labor_cost'],
            'overhead_cost': template['overhead_cost'],
            'total_cost': template['total_cost']
        }
        df = pd.DataFrame(data)
        # Add organic material costs
        organic_data = {
            'product_type': [product_class] * len(template['weight']),
            'material_type': ['organic'] * len(template['weight']),
            'weight': [w * 1.2 for w in template['weight']],  # Assuming a 20% weight increase
            'production_time': template['production_time'],
            'labor_cost': [l * 1.5 for l in template['labor_cost']],  # Assuming 50% increase
            'overhead_cost': [o * 1.5 for o in template['overhead_cost']],  # Assuming 50% increase
            'total_cost': [t * 1.5 for t in template['total_cost']]  # Assuming 50% increase
        }
        df_organic = pd.DataFrame(organic_data)
        return pd.concat([df, df_organic], ignore_index=True)
    else:
        print(f"Template for product type '{product_class}' not found.")
        return pd.DataFrame()

# Encode categorical variables
def encode_data(df):
    X = df[['product_type', 'material_type', 'weight', 'production_time', 'labor_cost', 'overhead_cost']]
    y = df['total_cost']
    X = pd.get_dummies(X, columns=['product_type', 'material_type'])
    return X, y

# Split the data
def split_and_scale_data(X, y):
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, scaler
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, X_scaled, y, y, scaler

# Train the model
def train_model(X_train, y_train):
    cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
    cost_model.fit(X_train, y_train)
    return cost_model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")

# Estimate cost
def estimate_cost(model, scaler, X, product_type, material_type, weight, production_time, labor_cost, overhead_cost, conversion_rate=320):
    input_data = pd.DataFrame([[product_type, material_type, weight, production_time, labor_cost, overhead_cost]], 
                              columns=['product_type', 'material_type', 'weight', 'production_time', 'labor_cost', 'overhead_cost'])
    input_data = pd.get_dummies(input_data, columns=['product_type', 'material_type'])
    
    # Ensure the input data has the same columns as the training data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder the columns to match the training data
    input_data = input_data[X.columns]
    
    input_data = scaler.transform(input_data)
    estimated_cost_usd = model.predict(input_data)[0]
    estimated_cost_lkr = estimated_cost_usd * conversion_rate
    return estimated_cost_usd, estimated_cost_lkr

# Define a Pydantic model for the incoming request
class EstimateCostRequest(BaseModel):
    img_url: str

@app.post("/estimate-cost")
async def estimate_cost_api(request: EstimateCostRequest):
    img_url = request.img_url

    # Classify the image using OpenAI Vision API
    product_class = classify_image(img_url)
    
    # Generate data based on detected product type
    df = generate_data(product_class)
    if not df.empty:
        # Encode, split, scale, and train the model
        X, y = encode_data(df)
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
        cost_model = train_model(X_train, y_train)
        if len(X_train) > 1:
            evaluate_model(cost_model, X_test, y_test)

        # Define the new material and update factors
        new_material = 'organic'
        weight_increase_factor = 1.2  # Example factor
        cost_increase_factor = 1.5  # Example factor

        # Update the DataFrame with the new material
        df.loc[df['product_type'] == product_class, 'material_type'] = new_material
        df.loc[df['product_type'] == product_class, 'weight'] *= weight_increase_factor
        df.loc[df['product_type'] == product_class, 'total_cost'] *= cost_increase_factor

        # Estimate the cost
        usd_cost, lkr_cost = estimate_cost(cost_model, scaler, X, product_class, new_material, 
                                           df.loc[df['product_type'] == product_class, 'weight'].values[0], 
                                           df.loc[df['product_type'] == product_class, 'production_time'].values[0], 
                                           df.loc[df['product_type'] == product_class, 'labor_cost'].values[0], 
                                           df.loc[df['product_type'] == product_class, 'overhead_cost'].values[0])

        response = {
            "product_type": product_class,
            "usd_cost": usd_cost,
            "lkr_cost": lkr_cost,
            "new_material_type": new_material
        }
    else:
        response = {
            "error": "Could not generate data for the detected product."
        }

    return response


