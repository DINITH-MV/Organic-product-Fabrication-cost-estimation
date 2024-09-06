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
        'labor_cost': [0.8, 1.0],  # USD
        'overhead_cost': [0.1, 0.15],  # USD
        'total_cost': [1.6, 2.5]  # USD (plastic vs. organic)
    },
    'container': {
        'weight': [0.2, 0.25],  # kg
        'production_time': [0.7, 0.8],  # hours
        'labor_cost': [1.167, 1.3],  # USD
        'overhead_cost': [0.2, 0.25],  # USD
        'total_cost': [2.0, 3.0]  # USD (plastic vs. organic)
    },
    'bag': {
        'weight': [0.05, 0.06],  # kg
        'production_time': [0.3, 0.35],  # hours
        'labor_cost': [0.3, 0.4],  # USD
        'overhead_cost': [0.05, 0.07],  # USD
        'total_cost': [0.6, 1.0]  # USD (plastic vs. organic)
    },
    'cup': {
        'weight': [0.15, 0.18],  # kg
        'production_time': [0.6, 0.7],  # hours
        'labor_cost': [1.0, 1.167],  # USD
        'overhead_cost': [0.15, 0.2],  # USD
        'total_cost': [2.0, 2.83]  # USD (plastic vs. organic)
    },
    'plate': {
        'weight': [0.1, 0.12],  # kg
        'production_time': [0.5, 0.6],  # hours
        'labor_cost': [0.8, 2.0],  # USD
        'overhead_cost': [0.1, 0.15],  # USD
        'total_cost': [1.6, 2.5]  # USD (plastic vs. organic)
    },
    'utensil': {
        'weight': [0.05, 0.06],  # kg
        'production_time': [0.4, 0.5],  # hours
        'labor_cost': [0.5, 0.58],  # USD
        'overhead_cost': [0.07, 0.1],  # USD
        'total_cost': [1.0, 1.33]  # USD (plastic vs. organic)
    },
    'tub': {
        'weight': [0.3, 0.35],  # kg
        'production_time': [0.8, 0.9],  # hours
        'labor_cost': [1.33, 1.5],  # USD
        'overhead_cost': [0.3, 0.35],  # USD
        'total_cost': [2.5, 3.66]  # USD (plastic vs. organic)
    },
    'wrap': {
        'weight': [0.02, 0.025],  # kg
        'production_time': [0.2, 0.25],  # hours
        'labor_cost': [0.167, 0.2],  # USD
        'overhead_cost': [0.02, 0.03],  # USD
        'total_cost': [0.3, 0.5]  # USD (plastic vs. organic)
    },
    'tray': {
        'weight': [0.25, 0.3],  # kg
        'production_time': [0.7, 0.8],  # hours
        'labor_cost': [1.167, 1.33],  # USD
        'overhead_cost': [0.2, 0.25],  # USD
        'total_cost': [2.0, 3.0]  # USD (plastic vs. organic)
    },
    'bucket': {
        'weight': [0.5, 0.6],  # kg
        'production_time': [1.0, 1.2],  # hours
        'labor_cost': [3.33, 4.167],  # USD
        'overhead_cost': [0.2, 0.3],  # USD
        'total_cost': [8.3, 11.667]  # USD (plastic vs. organic)
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

# Estimate cost directly without training any model
def estimate_cost(df, conversion_rate=320):
    # Assuming the DataFrame `df` contains the relevant data for cost calculation
    usd_cost = df['total_cost'].mean()  # Average total cost in USD
    lkr_cost = usd_cost * conversion_rate  # Convert to LKR using a fixed rate
    return usd_cost, lkr_cost

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
        # Estimate the cost directly
        usd_cost, lkr_cost = estimate_cost(df)

        response = {
            "product_type": product_class,
            "usd_cost": usd_cost,
            "lkr_cost": lkr_cost,
            "new_material_type": "organic"
        }
    else:
        response = {
            "error": "Could not generate data for the detected product."
        }

    return response


