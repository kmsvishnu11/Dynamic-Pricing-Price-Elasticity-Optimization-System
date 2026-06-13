import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from src.elasticity import fit_demand_curve

def find_optimal_price(predict_demand_func, cost, price_range=None):
    """Find revenue and profit maximizing prices."""
    if price_range is None:
        # Default search range
        search_min = cost * 1.1 if cost > 0 else 0.01
        search_max = 1000 # fallback
    else:
        search_min, search_max = price_range

    # Revenue maximizing
    def neg_revenue(p):
        q = predict_demand_func(p)
        return -(p * q)
    
    res_rev = minimize_scalar(neg_revenue, bounds=(search_min, search_max), method='bounded')
    rev_opt_p = res_rev.x
    expected_rev = -res_rev.fun
    
    # Profit maximizing
    def neg_profit(p):
        q = predict_demand_func(p)
        return -((p - cost) * q)
        
    res_prof = minimize_scalar(neg_profit, bounds=(search_min, search_max), method='bounded')
    prof_opt_p = res_prof.x
    expected_prof = -res_prof.fun
    
    return rev_opt_p, prof_opt_p, expected_rev, expected_prof

def simulate_price_change(predict_demand_func, current_price, new_price, current_cost):
    """Simulate the impact of a price change."""
    q_curr = predict_demand_func(current_price)
    q_new = predict_demand_func(new_price)
    
    rev_curr = current_price * q_curr
    rev_new = new_price * q_new
    
    prof_curr = (current_price - current_cost) * q_curr
    prof_new = (new_price - current_cost) * q_new
    
    return {
        'current_quantity': q_curr,
        'new_quantity': q_new,
        'current_revenue': rev_curr,
        'new_revenue': rev_new,
        'current_profit': prof_curr,
        'new_profit': prof_new,
        'revenue_change_pct': (rev_new - rev_curr) / rev_curr * 100 if rev_curr != 0 else 0,
        'demand_change_pct': (q_new - q_curr) / q_curr * 100 if q_curr != 0 else 0
    }

def dynamic_price_adjust(base_price, demand_multiplier, inventory_level, competitor_price, elasticity):
    """Apply dynamic rules to adjust price."""
    price = base_price
    breakdown = {}
    
    # Time-based
    if demand_multiplier > 1.2:
        adj = 1.10
        price *= adj
        breakdown['time_adjustment'] = f"+{int((adj-1)*100)}% (High Demand)"
    elif demand_multiplier < 0.8:
        adj = 0.92
        price *= adj
        breakdown['time_adjustment'] = f"{int((adj-1)*100)}% (Low Demand)"
    
    # Inventory-based
    if inventory_level < 0.2: # < 20%
        adj = 1.10
        price *= adj
        breakdown['inventory_adjustment'] = "+10% (Scarcity)"
    elif inventory_level > 0.8: # > 80%
        adj = 0.95
        price *= adj
        breakdown['inventory_adjustment'] = "-5% (Clearance)"
        
    # Competitor-based
    if competitor_price < price * 0.9:
        price = max(competitor_price, price * 0.9) # Don't drop too much
        breakdown['competitor_adjustment'] = "Price match (Limited)"
    elif competitor_price > price * 1.1:
        price *= 1.05 # Opportunity for small premium
        breakdown['competitor_adjustment'] = "+5% (Premium Opportunity)"
        
    # Elasticity-based bounds
    max_change = 0.05 if elasticity < -1.5 else 0.15
    lower_bound = base_price * (1 - max_change)
    upper_bound = base_price * (1 + max_change)
    
    final_price = max(lower_bound, min(upper_bound, price))
    if final_price != price:
        breakdown['elasticity_constraint'] = "Adjustment capped by elasticity sensitivity"
        
    return final_price, breakdown

def generate_price_recommendations(price_qty_df, elasticity_df, cost_assumption_pct=0.7):
    """Generate recommendations for all products."""
    recommendations = []
    
    for _, row in elasticity_df.iterrows():
        sc = row['StockCode']
        prod_data = price_qty_df[price_qty_df['StockCode'] == sc]
        curr_p = prod_data['Price'].mean()
        cost = curr_p * cost_assumption_pct
        
        _, _, _, predict_func = fit_demand_curve(price_qty_df, sc)
        if predict_func:
            rev_opt_p, prof_opt_p, _, _ = find_optimal_price(predict_func, cost, price_range=(curr_p*0.5, curr_p*2.0))
            
            # Recommendation based on profit optimal
            diff_pct = (prof_opt_p - curr_p) / curr_p
            if diff_pct > 0.05:
                rec = f"Increase by {diff_pct*100:.1f}%"
            elif diff_pct < -0.05:
                rec = f"Decrease by {abs(diff_pct)*100:.1f}%"
            else:
                rec = "Price is optimal"
                
            recommendations.append({
                'StockCode': sc,
                'current_price': curr_p,
                'optimal_profit_price': prof_opt_p,
                'recommendation': rec
            })
            
    return pd.DataFrame(recommendations)

def revenue_simulation(predict_demand_func, price_range, n_simulations=1000):
    """Monte Carlo simulation for revenue with demand uncertainty."""
    prices = np.linspace(price_range[0], price_range[1], 50)
    all_revs = []
    
    for p in prices:
        base_q = predict_demand_func(p)
        # Random noise ±10%
        sim_qs = base_q * (1 + np.random.uniform(-0.1, 0.1, n_simulations))
        sim_revs = p * sim_qs
        all_revs.append(sim_revs)
        
    all_revs = np.array(all_revs)
    mean_rev = np.mean(all_revs, axis=1)
    lower_95 = np.percentile(all_revs, 2.5, axis=1)
    upper_95 = np.percentile(all_revs, 97.5, axis=1)
    
    return prices, mean_rev, lower_95, upper_95
