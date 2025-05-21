# ------------------------------------------------------------
# Dashboard integral de ventas — Streamlit
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
from packaging import version
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- Configuración de página ----------
st.set_page_config(page_title="Dashboard Ventas", layout="wide")

# ---------- Carga y cacheo de datos ----------
if version.parse(st.__version__) >= version.parse("1.18"):
    cache = st.cache_data
else:
    cache = st.cache

@cache
def load_data(path="data.csv"):
    # Asegúrate de que data.csv esté en la misma carpeta que este script
    return pd.read_csv(path, parse_dates=['Date'])

df = load_data()

# ---------- Filtros en sidebar ----------
st.sidebar.header("Filtros")
ciudades = st.sidebar.multiselect(
    "Ciudad", df['City'].unique(), df['City'].unique()
)
lineas = st.sidebar.multiselect(
    "Línea de producto", df['Product line'].unique(), df['Product line'].unique()
)

df_filt = df[
    df['City'].isin(ciudades) & df['Product line'].isin(lineas)
]

# ---------- Pestañas del dashboard ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Panorama temporal",
    "Detalle producto",
    "Correlaciones",
    "Patrones multivariados",
    "Vista 3D"
])

# =====================================================
# TAB 1 · PANORAMA TEMPORAL
# =====================================================
with tab1:
    st.subheader("Ventas diarias")
    serie = df_filt.groupby('Date')['Total'].sum().reset_index()
    st.line_chart(serie, x='Date', y='Total')

    st.subheader("Ventas mensuales vs rating medio")
    mensual = (
        df_filt.set_index('Date')
               .resample('ME')                 # Month-End
               .agg({'Total': 'sum', 'Rating': 'mean'})
               .reset_index()
    )

    # Barras + línea en la misma figura
    fig_bars = px.bar(mensual, x='Date', y='Total', labels={'Total':'Ventas (USD)'})
    fig_bars.add_scatter(
        x=mensual['Date'], y=mensual['Rating'],
        mode='lines+markers', name='Rating medio', yaxis='y2'
    )
    fig_bars.update_layout(
        yaxis2=dict(overlaying='y', side='right', title='Rating medio'),
        title="Ventas mensuales y satisfacción del cliente"
    )
    st.plotly_chart(fig_bars, use_container_width=True)

# =====================================================
# TAB 2 · DETALLE PRODUCTO
# =====================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precio vs cantidad")
        st.plotly_chart(
            px.scatter(
                df_filt, x='Unit price', y='Quantity',
                color='Product line', hover_data=['Total']
            ),
            use_container_width=True
        )

    with col2:
        st.subheader("Ingresos por línea de producto")
        st.plotly_chart(
            px.box(df_filt, x='Product line', y='gross income', color='Product line'),
            use_container_width=True
        )

# =====================================================
# TAB 3 · CORRELACIONES
# =====================================================
with tab3:
    st.subheader("Mapa de calor de correlaciones (variables numéricas)")
    corr = df_filt.select_dtypes('number').corr().fillna(0)
    st.plotly_chart(
        px.imshow(
            corr, text_auto='.2f', aspect='auto',
            color_continuous_scale='RdBu', origin='lower'
        ),
        use_container_width=True
    )

# =====================================================
# TAB 4 · PATRONES MULTIVARIADOS
# =====================================================
with tab4:
    st.subheader("Coordenadas paralelas por ciudad")

    # Convertir la ciudad a código numérico para usarlo como color
    subset = df_filt[['Unit price', 'Quantity', 'gross income', 'Rating', 'City']].copy()
    subset['City_code'] = pd.Categorical(subset['City']).codes

    fig_pc = px.parallel_coordinates(
        subset,
        dimensions=['Unit price', 'Quantity', 'gross income', 'Rating'],
        color='City_code',
        color_continuous_scale='Turbo',
        labels={'City_code': 'Ciudad'}
    )
    st.plotly_chart(fig_pc, use_container_width=True)

    st.divider()

    st.subheader("PCA — agrupación por línea de producto")
    num_cols = ['Unit price', 'Quantity', 'Total', 'gross income', 'Rating']
    X = StandardScaler().fit_transform(df_filt[num_cols])
    pca_res = PCA(n_components=2).fit_transform(X)
    pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
    pca_df['Product line'] = df_filt['Product line'].values

    st.plotly_chart(
        px.scatter(
            pca_df, x='PC1', y='PC2', color='Product line',
            title="Proyección 2D de PCA"
        ),
        use_container_width=True
    )

# =====================================================
# TAB 5 · VISTA 3D
# =====================================================
with tab5:
    st.subheader("Precio – Cantidad – Ingreso en 3D")
    st.plotly_chart(
        px.scatter_3d(
            df_filt, x='Unit price', y='Quantity', z='gross income',
            color='Product line', height=700
        ),
        use_container_width=True
    )
