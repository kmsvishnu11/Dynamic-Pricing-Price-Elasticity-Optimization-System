from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.elasticity import fit_demand_curve
from src.pricing import find_optimal_price, simulate_price_change

app = FastAPI(title="Dynamic Pricing API")

# Global variables for data (in production, use a database/cache)
product_agg = None
price_demand = None
segments = None

def load_data():
    global product_agg, price_demand, segments
    try:
        product_agg = pd.read_pickle('data/product_agg.pkl')
        price_demand = pd.read_pickle('data/price_demand.pkl')
        segments = pd.read_pickle('data/segments.pkl')
    except Exception as e:
        print(f"Error loading precomputed data: {e}")

load_data()

class PriceOptimizationRequest(BaseModel):
    product_id: str
    cost: float
    min_price: float = None
    max_price: float = None

class SimulationRequest(BaseModel):
    product_id: str
    current_price: float
    new_price: float
    cost: float

@app.get("/")
def read_root():
    return {"message": "Dynamic Pricing API is running"}

@app.get("/products")
def get_products():
    if product_agg is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    return product_agg[['StockCode', 'Description', 'price_mean', 'total_revenue']].head(100).to_dict(orient='records')

@app.post("/optimize")
def optimize_price(req: PriceOptimizationRequest):
    if price_demand is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    _, _, _, predict_func = fit_demand_curve(price_demand, req.product_id)
    if not predict_func:
        raise HTTPException(status_code=404, detail="Product not found or not enough data for modeling")
    
    curr_p = product_agg[product_agg['StockCode'] == req.product_id]['price_mean'].values[0]
    p_range = (req.min_price or curr_p * 0.5, req.max_price or curr_p * 2.0)
    
    rev_p, prof_p, exp_rev, exp_prof = find_optimal_price(predict_func, req.cost, price_range=p_range)
    
    return {
        "product_id": req.product_id,
        "revenue_maximizing_price": rev_p,
        "profit_maximizing_price": prof_p,
        "expected_revenue": exp_rev,
        "expected_profit": exp_prof
    }

@app.post("/simulate")
def simulate_price(req: SimulationRequest):
    if price_demand is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    _, _, _, predict_func = fit_demand_curve(price_demand, req.product_id)
    if not predict_func:
        raise HTTPException(status_code=404, detail="Product not found")
        
    results = simulate_price_change(predict_func, req.current_price, req.new_price, req.cost)
    return results

@app.get("/health")
def health_check():
    return {"status": "healthy", "data_loaded": product_agg is not None}
