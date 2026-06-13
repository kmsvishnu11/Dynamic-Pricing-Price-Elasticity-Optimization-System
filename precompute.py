import pandas as pd
import os
from src.preprocess import load_data, aggregate_product_data, create_price_demand_pairs, add_time_features
from src.elasticity import compute_elasticity_all_products, segment_products_by_elasticity

def run_precalculation():
    print("Loading data...")
    df = load_data('data/online_retail.csv')
    
    print("Adding time features...")
    df = add_time_features(df)
    
    print("Aggregating product data...")
    product_agg = aggregate_product_data(df)
    
    print("Creating price-demand pairs...")
    price_demand = create_price_demand_pairs(df)
    # Filter price_demand to only include products in product_agg
    price_demand = price_demand[price_demand['StockCode'].isin(product_agg['StockCode'])]
    
    print("Computing elasticity for all products...")
    elasticity_df = compute_elasticity_all_products(price_demand)
    
    print("Segmenting products...")
    segments_df = segment_products_by_elasticity(elasticity_df, product_agg)
    
    # Save processed data
    print("Saving processed data...")
    product_agg.to_pickle('data/product_agg.pkl')
    price_demand.to_pickle('data/price_demand.pkl')
    segments_df.to_pickle('data/segments.pkl')
    print("Done!")

if __name__ == "__main__":
    run_precalculation()
