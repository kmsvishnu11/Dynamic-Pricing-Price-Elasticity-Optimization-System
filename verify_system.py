import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

from src.preprocess import load_data
from src.elasticity import fit_demand_curve
from src.pricing import find_optimal_price

def test_system():
    print("--- Starting System Verification ---")
    
    # 1. Test Data Loading
    try:
        df = pd.read_pickle('data/price_demand.pkl')
        product_agg = pd.read_pickle('data/product_agg.pkl')
        print(f"✅ Data loaded. {len(product_agg)} products available.")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return

    # 2. Test Elasticity & Demand Fitting
    try:
        test_product = product_agg.iloc[0]['StockCode']
        print(f"Testing product: {test_product}")
        
        model, method, r2, predict_func = fit_demand_curve(df, test_product)
        if predict_func:
            print(f"✅ Demand curve fitted using {method} (R2: {r2:.3f})")
            q_test = predict_func(product_agg.iloc[0]['price_mean'])
            print(f"✅ Prediction test: Predicted Q at mean price: {q_test:.2f}")
        else:
            print("❌ Demand curve fitting failed.")
            return
    except Exception as e:
        print(f"❌ Elasticity test failed: {e}")
        return

    # 3. Test Optimization
    try:
        curr_p = product_agg.iloc[0]['price_mean']
        cost = curr_p * 0.7
        rev_p, prof_p, exp_rev, exp_prof = find_optimal_price(predict_func, cost, price_range=(curr_p*0.5, curr_p*2.0))
        print(f"✅ Optimization successful:")
        print(f"   - Revenue-Max Price: {rev_p:.2f}")
        print(f"   - Profit-Max Price: {prof_p:.2f}")
        print(f"   - Expected Profit: {exp_prof:.2f}")
    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
        return

    print("--- System Verification Complete: ALL PASSED ---")

if __name__ == "__main__":
    test_system()
