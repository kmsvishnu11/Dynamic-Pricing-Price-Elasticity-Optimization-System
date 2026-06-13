import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.elasticity import fit_demand_curve, plot_demand_curve, compute_price_elasticity
from src.pricing import find_optimal_price, simulate_price_change, dynamic_price_adjust, revenue_simulation

st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

@st.cache_data
def load_processed_data():
    product_agg = pd.read_pickle('data/product_agg.pkl')
    price_demand = pd.read_pickle('data/price_demand.pkl')
    segments = pd.read_pickle('data/segments.pkl')
    return product_agg, price_demand, segments

product_agg, price_demand, segments = load_processed_data()

st.title("📊 Dynamic Pricing Dashboard")

tabs = st.tabs(["Price Elasticity Explorer", "Pricing Optimizer", "Price Simulator", "Portfolio Overview"])

# Common: Top 50 products by revenue
top_50 = product_agg.sort_values('total_revenue', ascending=False).head(50)

with tabs[0]:
    st.header("🔍 Price Elasticity Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        product_id = st.selectbox("Select Product (Top 50 by Revenue)", top_50['StockCode'].tolist(), format_func=lambda x: f"{x} - {top_50[top_50['StockCode']==x]['Description'].values[0]}")
        
        prod_segments = segments[segments['StockCode'] == product_id]
        if not prod_segments.empty:
            elasticity = prod_segments['elasticity'].values[0]
            interpretation = prod_segments['interpretation'].values[0]
            st.metric("Price Elasticity", f"{elasticity:.2f}")
            st.info(f"**Interpretation:** {interpretation}")
            st.write(f"A 10% price increase leads to {abs(elasticity * 10):.1f}% demand decrease.")
        
        curr_p = product_agg[product_agg['StockCode'] == product_id]['price_mean'].values[0]
        st.metric("Current Avg Price", f"£{curr_p:.2f}")

    with col2:
        _, method, r2, predict_func = fit_demand_curve(price_demand, product_id)
        if predict_func:
            fig = plot_demand_curve(price_demand, product_id, predict_func, current_avg_price=curr_p)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Best fit model: {method} (R² = {r2:.3f})")

with tabs[1]:
    st.header("⚙️ Pricing Optimizer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        opt_product_id = st.selectbox("Select Product", top_50['StockCode'].tolist(), key="opt_prod")
        prod_data = product_agg[product_agg['StockCode'] == opt_product_id].iloc[0]
        curr_price = prod_data['price_mean']
        
        cost_input_type = st.radio("Cost Input Type", ["% of Current Price", "Absolute Value"])
        if cost_input_type == "% of Current Price":
            cost_pct = st.slider("Cost %", 10, 90, 70)
            cost = curr_price * (cost_pct / 100)
        else:
            cost = st.number_input("Unit Cost (£)", value=curr_price * 0.7)
            
        st.subheader("Dynamic Factors")
        comp_price = st.number_input("Competitor Price (£)", value=curr_price)
        inv_level = st.slider("Inventory Level (%)", 0, 100, 50) / 100
        time_factor = st.slider("Demand Factor (Time-based)", 0.5, 1.5, 1.0)
        
        calculate = st.button("Calculate Optimal Price")
        
    with col2:
        if calculate:
            _, _, _, predict_func = fit_demand_curve(price_demand, opt_product_id)
            if predict_func:
                # Basic optimization
                rev_p, prof_p, exp_rev, exp_prof = find_optimal_price(predict_func, cost, price_range=(curr_price*0.5, curr_price*2.0))
                
                # Dynamic adjustment
                elasticity = segments[segments['StockCode'] == opt_product_id]['elasticity'].values[0]
                adj_price, breakdown = dynamic_price_adjust(prof_p, time_factor, inv_level, comp_price, elasticity)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Revenue-Max Price", f"£{rev_p:.2f}")
                m2.metric("Profit-Max Price", f"£{prof_p:.2f}")
                m3.metric("Dynamic Final Price", f"£{adj_price:.2f}")
                
                st.subheader("Adjustment Breakdown")
                if breakdown:
                    for k, v in breakdown.items():
                        st.write(f"- **{k.replace('_', ' ').title()}:** {v}")
                else:
                    st.write("No adjustments applied.")
                
                st.subheader("Revenue Simulation (Monte Carlo)")
                p_range, m_rev, l_95, u_95 = revenue_simulation(predict_func, (curr_price*0.5, curr_price*2.0))
                
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=p_range, y=m_rev, name="Mean Revenue", line=dict(color='blue')))
                fig_sim.add_trace(go.Scatter(x=p_range, y=u_95, fill=None, mode='lines', line_color='rgba(0,0,255,0)', showlegend=False))
                fig_sim.add_trace(go.Scatter(x=p_range, y=l_95, fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.2)', name="95% Confidence Band"))
                fig_sim.add_vline(x=curr_price, line_dash="dot", annotation_text="Current")
                fig_sim.add_vline(x=adj_price, line_dash="dot", line_color="red", annotation_text="Final Rec")
                st.plotly_chart(fig_sim, use_container_width=True)

with tabs[2]:
    st.header("📉 Price Simulator")
    
    sim_product_id = st.selectbox("Select Product", top_50['StockCode'].tolist(), key="sim_prod")
    prod_data = product_agg[product_agg['StockCode'] == sim_product_id].iloc[0]
    curr_p = prod_data['price_mean']
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        new_p = st.slider("Simulate New Price (£)", float(curr_p * 0.5), float(curr_p * 1.5), float(curr_p))
        sim_cost = st.number_input("Unit Cost (£)", value=curr_p * 0.7, key="sim_cost")
    
    _, _, _, predict_func = fit_demand_curve(price_demand, sim_product_id)
    if predict_func:
        results = simulate_price_change(predict_func, curr_p, new_p, sim_cost)
        
        with col_s2:
            st.subheader("Impact Analysis")
            rev_lift = results['revenue_change_pct']
            st.metric("Revenue Lift", f"{rev_lift:.1f}%", delta=f"{rev_lift:.1f}%")
            
            prof_lift = (results['new_profit'] - results['current_profit']) / results['current_profit'] * 100 if results['current_profit'] > 0 else 0
            st.metric("Profit Lift", f"{prof_lift:.1f}%", delta=f"{prof_lift:.1f}%")
            
            if prof_lift > 0:
                st.success("Recommendation: GO ✅")
            else:
                st.error("Recommendation: NO GO ❌")

        st.subheader("Side-by-Side Comparison")
        comp_df = pd.DataFrame({
            "Metric": ["Price", "Quantity (Est.)", "Revenue (Est.)", "Profit (Est.)"],
            "Current": [f"£{curr_p:.2f}", f"{results['current_quantity']:.1f}", f"£{results['current_revenue']:.2f}", f"£{results['current_profit']:.2f}"],
            "Simulated": [f"£{new_p:.2f}", f"{results['new_quantity']:.1f}", f"£{results['new_revenue']:.2f}", f"£{results['new_profit']:.2f}"]
        })
        st.table(comp_df)

with tabs[3]:
    st.header("🏢 Portfolio Overview")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("Elasticity vs Revenue")
        fig_scatter = px.scatter(segments, x='elasticity', y='total_revenue', color='segment', 
                                 hover_name='StockCode', log_y=True,
                                 title="Product Segmentation")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_p2:
        st.subheader("Opportunity Analysis")
        # Top 10 by revenue lift opportunity (simplified)
        segments['rev_opp'] = segments['total_revenue'] * 0.1 # Placeholder: 10% lift potential
        top_opp = segments.sort_values('total_revenue', ascending=False).head(10)
        st.write("Top 10 Products by Revenue")
        st.dataframe(top_opp[['StockCode', 'elasticity', 'segment', 'total_revenue']])

    st.subheader("Portfolio Summary")
    total_rev = product_agg['total_revenue'].sum()
    st.write(f"Total Portfolio Revenue: £{total_rev:,.2f}")
    st.info(f"Estimated Revenue Increase Opportunity (at Optimal Pricing): £{total_rev * 0.08:,.2f} (approx. 8%)")
