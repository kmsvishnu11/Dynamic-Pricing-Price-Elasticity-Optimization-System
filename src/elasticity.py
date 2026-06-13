import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar

def compute_price_elasticity(price_qty_df, product_id):
    """Compute price elasticity for a single product using OLS log-log regression."""
    product_df = price_qty_df[price_qty_df['StockCode'] == product_id].copy()
    if len(product_df) < 2:
        return None, 0, "Not enough data", None
    
    # log-log model: log(Q) = a + b * log(P)
    # b is the elasticity
    X = np.log(product_df['Price'])
    y = np.log(product_df['Quantity'])
    X = sm.add_constant(X)
    
    try:
        model = sm.OLS(y, X).fit()
        elasticity = model.params.iloc[1]
        r_squared = model.rsquared
        
        if elasticity < -1:
            interpretation = "Elastic (Price Sensitive)"
        elif -1 <= elasticity < 0:
            interpretation = "Inelastic (Price Insensitive)"
        else:
            interpretation = "Positive Elasticity (Anomalous)"
            
        return elasticity, r_squared, interpretation, model
    except:
        return None, 0, "Model error", None

def fit_demand_curve(price_qty_df, product_id, method='auto'):
    """Fit demand curve using different models and select the best by R-squared."""
    product_df = price_qty_df[price_qty_df['StockCode'] == product_id].copy()
    if len(product_df) < 2:
        return None, None, 0, None
    
    results = []
    
    # 1. Log-Log: Q = a * P^b  => log(Q) = log(a) + b * log(P)
    try:
        X1 = sm.add_constant(np.log(product_df['Price']))
        y1 = np.log(product_df['Quantity'])
        model1 = sm.OLS(y1, X1).fit()
        results.append({'method': 'log-log', 'r2': model1.rsquared, 'model': model1})
    except: pass
    
    # 2. Linear: Q = a + b * P
    try:
        X2 = sm.add_constant(product_df['Price'])
        y2 = product_df['Quantity']
        model2 = sm.OLS(y2, X2).fit()
        results.append({'method': 'linear', 'r2': model2.rsquared, 'model': model2})
    except: pass
    
    # 3. Log-Linear: Q = a * e^(b*P) => log(Q) = log(a) + b * P
    try:
        X3 = sm.add_constant(product_df['Price'])
        y3 = np.log(product_df['Quantity'])
        model3 = sm.OLS(y3, X3).fit()
        results.append({'method': 'log-linear', 'r2': model3.rsquared, 'model': model3})
    except: pass
    
    if not results:
        return None, None, 0, None
        
    best = max(results, key=lambda x: x['r2'])
    
    def predict_demand(price):
        if best['method'] == 'log-log':
            return np.exp(best['model'].predict([1, np.log(price)]))[0]
        elif best['method'] == 'linear':
            return best['model'].predict([1, price])[0]
        elif best['method'] == 'log-linear':
            return np.exp(best['model'].predict([1, price]))[0]
            
    return best['model'], best['method'], best['r2'], predict_demand

def plot_demand_curve(price_qty_df, product_id, predict_demand_func, current_avg_price=None, optimal_price=None):
    """Plot actual price vs quantity and overlay fitted demand curve."""
    product_df = price_qty_df[price_qty_df['StockCode'] == product_id].copy()
    
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=product_df['Price'], 
        y=product_df['Quantity'],
        mode='markers',
        name='Actual Demand'
    ))
    
    # Fitted curve
    p_range = np.linspace(product_df['Price'].min() * 0.5, product_df['Price'].max() * 1.5, 100)
    q_fitted = [predict_demand_func(p) for p in p_range]
    
    fig.add_trace(go.Scatter(
        x=p_range, 
        y=q_fitted,
        mode='lines',
        name='Fitted Demand Curve',
        line=dict(dash='dash')
    ))
    
    if current_avg_price:
        fig.add_vline(x=current_avg_price, line_dash="dot", line_color="green", annotation_text="Current")
        
    if optimal_price:
        fig.add_vline(x=optimal_price, line_dash="dot", line_color="red", annotation_text="Optimal")
        
    fig.update_layout(title=f"Demand Curve for {product_id}", xaxis_title="Price", yaxis_title="Quantity")
    return fig

def compute_elasticity_all_products(price_demand_df):
    """Compute elasticity for all products and classify."""
    results = []
    stock_codes = price_demand_df['StockCode'].unique()
    
    for sc in stock_codes:
        e, r2, interp, _ = compute_price_elasticity(price_demand_df, sc)
        if e is not None:
            # Classification
            if e < -1.5: class_label = 'Highly Elastic'
            elif e < -1: class_label = 'Elastic'
            elif e < -0.5: class_label = 'Inelastic'
            else: class_label = 'Highly Inelastic'
            
            results.append({
                'StockCode': sc,
                'elasticity': e,
                'r2': r2,
                'interpretation': interp,
                'class': class_label
            })
            
    return pd.DataFrame(results)

def segment_products_by_elasticity(elasticity_df, product_agg_df):
    """Segment products into 4 quadrants based on elasticity and revenue."""
    merged = pd.merge(elasticity_df, product_agg_df[['StockCode', 'total_revenue']], on='StockCode')
    
    med_revenue = merged['total_revenue'].median()
    
    def get_segment(row):
        is_high_revenue = row['total_revenue'] > med_revenue
        is_inelastic = row['elasticity'] > -1
        
        if is_high_revenue and is_inelastic:
            return "Quadrant 1: Premium (High Rev, Inelastic)"
        elif is_high_revenue and not is_inelastic:
            return "Quadrant 2: Competitive (High Rev, Elastic)"
        elif not is_high_revenue and is_inelastic:
            return "Quadrant 3: Bundle (Low Rev, Inelastic)"
        else:
            return "Quadrant 4: Discount (Low Rev, Elastic)"
            
    merged['segment'] = merged.apply(get_segment, axis=1)
    return merged
