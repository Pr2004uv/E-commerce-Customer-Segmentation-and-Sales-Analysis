import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime as dt

# --- Page Configuration ---
st.set_page_config(
    page_title="E-commerce Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """Loads, cleans, and preprocesses the e-commerce data."""
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # --- Data Cleaning ---
    # Drop rows with missing CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)
    
    # Remove cancelled orders (InvoiceNo starts with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Remove rows with negative or zero quantity/price
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # --- Feature Engineering ---
    # Calculate TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Convert CustomerID to string
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df

# Load the data
try:
    df = load_data('Online Retail.xlsx')
except FileNotFoundError:
    st.error("Error: 'Online Retail.xlsx' not found. Please download the file and place it in the same directory.")
    st.stop()


# --- Sidebar Filters ---
st.sidebar.header("Filters")
country = st.sidebar.multiselect(
    "Select Country",
    options=df['Country'].unique(),
    default=df['Country'].unique()
)

# Filter data based on selection
df_selection = df[df['Country'].isin(country)]

# --- Main Page ---
st.title("🛒 E-commerce Sales & Customer Analysis Dashboard")
st.markdown("---")

# --- Key Performance Indicators (KPIs) ---
total_sales = int(df_selection['TotalPrice'].sum())
total_orders = df_selection['InvoiceNo'].nunique()
total_customers = df_selection['CustomerID'].nunique()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Sales", value=f"${total_sales:,}")
with col2:
    st.metric(label="Total Orders", value=f"{total_orders:,}")
with col3:
    st.metric(label="Total Customers", value=f"{total_customers:,}")

st.markdown("---")


# --- Visualizations ---

# 1. Monthly Sales Trend
st.subheader("Monthly Sales Trend")
df_selection['YearMonth'] = df_selection['InvoiceDate'].dt.to_period('M').astype(str)
monthly_sales = df_selection.groupby('YearMonth')['TotalPrice'].sum().reset_index()

fig_monthly_sales = px.line(
    monthly_sales,
    x='YearMonth',
    y='TotalPrice',
    title='Total Sales Over Time',
    labels={'YearMonth': 'Month', 'TotalPrice': 'Total Sales ($)'},
    markers=True
)
st.plotly_chart(fig_monthly_sales, use_container_width=True)


# 2. Sales by Country and Top Products
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Country")
    sales_by_country = df_selection.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).reset_index()
    fig_country_sales = px.bar(
        sales_by_country.head(10),
        x='Country',
        y='TotalPrice',
        title='Top 10 Countries by Sales',
        labels={'Country': 'Country', 'TotalPrice': 'Total Sales ($)'},
        color='Country'
    )
    st.plotly_chart(fig_country_sales, use_container_width=True)

with col2:
    st.subheader("Top Selling Products")
    top_products = df_selection.groupby('Description')['Quantity'].sum().sort_values(ascending=False).reset_index()
    fig_top_products = px.bar(
        top_products.head(10),
        x='Quantity',
        y='Description',
        orientation='h',
        title='Top 10 Products by Quantity Sold',
        labels={'Description': 'Product', 'Quantity': 'Total Quantity'},
        color='Description'
    )
    st.plotly_chart(fig_top_products, use_container_width=True)

st.markdown("---")


# --- RFM Analysis & Customer Segmentation ---
st.header("Customer Segmentation with RFM Analysis")

# Calculate RFM metrics
snapshot_date = df_selection['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = df_selection.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'MonetaryValue'}, inplace=True)

# Preprocess for K-Means
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# K-Means Clustering
st.sidebar.header("RFM Segmentation")
n_clusters = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Display RFM segments
st.subheader(f"Customer Segments (K={n_clusters})")
fig_rfm = px.scatter_3d(
    rfm,
    x='Recency',
    y='Frequency',
    z='MonetaryValue',
    color=rfm['Cluster'].astype(str),
    title='RFM Customer Segments',
    labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Orders)', 'MonetaryValue': 'Monetary Value ($)'}
)
fig_rfm.update_traces(marker=dict(size=5))
st.plotly_chart(fig_rfm, use_container_width=True)

st.info("""
