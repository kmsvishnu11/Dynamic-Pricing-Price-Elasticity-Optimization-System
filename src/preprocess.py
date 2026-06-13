import pandas as pd
import numpy as np

def load_data(path):
    """Load and clean the dataset."""
    df = pd.read_csv(path, encoding='latin-1')
    
    # Drop rows where Quantity <= 0 (returns)
    df = df[df['Quantity'] > 0]
    
    # Drop rows where Price <= 0
    df = df[df['Price'] > 0]
    
    # Drop rows with missing CustomerID
    df = df.dropna(subset=['Customer ID'])
    
    # Parse InvoiceDate as datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Add Revenue column
    df['Revenue'] = df['Quantity'] * df['Price']
    
    return df

def aggregate_product_data(df):
    """Aggregate data by product."""
    agg_funcs = {
        'Price': ['nunique', 'min', 'max', 'mean', 'std'],
        'Quantity': ['mean', 'std', 'sum'],
        'Revenue': 'sum',
        'Invoice': 'nunique'
    }
    
    # We need Description as well, but one StockCode might have multiple descriptions.
    # We'll take the most recent or common one.
    product_info = df.groupby('StockCode').agg({
        'Description': 'first',
        'Price': ['nunique', 'min', 'max', 'mean', 'std'],
        'Quantity': ['mean', 'std', 'sum'],
        'Revenue': 'sum',
        'Invoice': 'nunique'
    })
    
    product_info.columns = [
        'Description', 'price_points_count', 'price_min', 'price_max', 'price_mean', 'price_std',
        'quantity_mean', 'quantity_std', 'total_quantity', 'total_revenue', 'transaction_count'
    ]
    
    # Filter products with at least 5 different price points
    product_info = product_info[product_info['price_points_count'] >= 5]
    
    # For price_points list and avg_quantity_per_price dict
    # This is quite heavy to do in a single agg, so we'll do it separately for the filtered products
    eligible_stock_codes = product_info.index.tolist()
    filtered_df = df[df['StockCode'].isin(eligible_stock_codes)]
    
    # Calculate avg_quantity_per_price: dict of price -> avg_qty
    # We group by StockCode and Price to get mean quantity at each price point
    price_demand = filtered_df.groupby(['StockCode', 'Price'])['Quantity'].mean().reset_index()
    
    def get_price_info(group):
        prices = group['Price'].tolist()
        qtys = group['Quantity'].tolist()
        return {
            'price_points': prices,
            'avg_quantity_per_price': dict(zip(prices, qtys))
        }
    
    extra_info = price_demand.groupby('StockCode').apply(get_price_info)
    
    # Merge back
    product_info['price_points'] = extra_info.apply(lambda x: x['price_points'])
    product_info['avg_quantity_per_price'] = extra_info.apply(lambda x: x['avg_quantity_per_price'])
    
    return product_info.reset_index()

def create_price_demand_pairs(df):
    """Create (price, quantity) pairs for each product and normalize."""
    # Group by StockCode + Price -> mean Quantity
    price_demand = df.groupby(['StockCode', 'Price', 'Description'])['Quantity'].mean().reset_index()
    
    # Normalize quantity per product
    def normalize(group):
        group['NormQuantity'] = group['Quantity'] / group['Quantity'].max()
        return group
    
    price_demand = price_demand.groupby('StockCode', group_keys=False).apply(normalize)
    
    return price_demand

def add_time_features(df):
    """Add time-based features and demand multipliers."""
    df = df.copy()
    df['hour'] = df['InvoiceDate'].dt.hour
    df['day'] = df['InvoiceDate'].dt.day
    df['month'] = df['InvoiceDate'].dt.month
    df['dayofweek'] = df['InvoiceDate'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
    
    # Add demand_multiplier based on time
    # Default is 1.0. Weekends and Holiday Season = higher demand.
    df['demand_multiplier'] = 1.0
    df.loc[df['is_weekend'] == 1, 'demand_multiplier'] += 0.2
    df.loc[df['is_holiday_season'] == 1, 'demand_multiplier'] += 0.3
    
    return df
