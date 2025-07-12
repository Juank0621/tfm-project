import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# Imports for RoBERTa sentiment analysis
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="🍽️ Restaurant Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍽️"
)

# ==========================================
# ESTILOS CSS MEJORADOS
# ==========================================
st.markdown("""
<style>
    /* Importar Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables de color */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #262730;
        --bg-tertiary: #1e1e2e;
        --sidebar-bg: #1a1d29;
        --text-primary: #fafafa;
        --text-secondary: #a0a0a0;
        --accent-blue: #4f46e5;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --border-color: #3d4043;
        --sidebar-border: #4f46e5;
    }
    
    /* Aplicación principal */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* SIDEBAR CON BORDE VISIBLE */
    .css-1d391kg {
        background-color: var(--sidebar-bg) !important;
        border-right: 3px solid var(--sidebar-border) !important;
        box-shadow: 2px 0 10px rgba(79, 70, 229, 0.1) !important;
    }
    
    /* Contenido principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: var(--bg-primary);
    }
    
    /* Títulos mejorados */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Tarjetas de métricas mejoradas */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-blue);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Elementos de formulario en sidebar */
    .stSelectbox > div > div {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    
    .stSlider > div > div > div {
        color: var(--accent-blue) !important;
    }
    
    /* Botones mejorados */
    .stButton > button {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Separadores */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* DataFrames */
    .dataframe {
        background-color: var(--bg-secondary) !important;
        border-radius: 8px !important;
    }
    
    /* Textos informativos */
    .info-box {
        background-color: var(--bg-secondary);
        border-left: 4px solid var(--accent-blue);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Sidebar título */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent-blue);
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FUNCIONES DE DATOS
# ==========================================

@st.cache_resource
def load_sentiment_model():
    """Load the RoBERTa sentiment analysis model"""
    if TRANSFORMERS_AVAILABLE:
        try:
            # Load the RoBERTa model for sentiment analysis
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            return sentiment_pipeline
        except Exception as e:
            st.error(f"Error loading RoBERTa model: {e}")
            return None
    else:
        st.warning("⚠️ Transformers library not available. Using simulated results.")
        return None

def analyze_sentiment_text(text, model):
    """Analyze sentiment of a text using RoBERTa model"""
    if model is None or not TRANSFORMERS_AVAILABLE:
        # Fallback to simulation if model not available
        sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        probs = np.random.dirichlet([0.6, 0.2, 0.2])
        return [{'label': sent, 'score': prob} for sent, prob in zip(sentiments, probs)]
    
    try:
        # Use the actual RoBERTa model
        results = model(text)
        # The model returns a list of dictionaries with 'label' and 'score'
        return results[0] if results else []
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        # Fallback to simulation
        sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        probs = np.random.dirichlet([0.6, 0.2, 0.2])
        return [{'label': sent, 'score': prob} for sent, prob in zip(sentiments, probs)]

@st.cache_data
def load_sample_data():
    """Generar datos de muestra para el dashboard"""
    np.random.seed(42)
    
    # Estados principales de USA con coordenadas
    states_data = {
        'state': ['CA', 'NY', 'TX', 'FL', 'PA', 'IL', 'OH', 'NC', 'MI', 'GA'],
        'state_name': ['California', 'New York', 'Texas', 'Florida', 'Pennsylvania', 
                      'Illinois', 'Ohio', 'North Carolina', 'Michigan', 'Georgia'],
        'lat': [36.7783, 40.7128, 31.9686, 27.7663, 40.2732, 41.8781, 40.4173, 35.7596, 42.3314, 33.7490],
        'lon': [-119.4179, -74.0060, -99.9018, -82.6404, -76.8839, -87.6298, -82.9071, -79.0193, -84.5555, -84.3880]
    }
    
    states_df = pd.DataFrame(states_data)
    
    # Generar datos de restaurantes
    restaurants = []
    cities = ['Los Angeles', 'New York', 'Houston', 'Miami', 'Philadelphia', 
             'Chicago', 'Columbus', 'Charlotte', 'Detroit', 'Atlanta']
    
    for i, (state, city, lat, lon) in enumerate(zip(states_data['state'], cities, 
                                                   states_data['lat'], states_data['lon'])):
        n_restaurants = np.random.randint(30, 80)  # Reducido de 50-200 a 30-80
        
        for j in range(n_restaurants):
            # Agregar variación aleatoria a las coordenadas
            rest_lat = lat + np.random.normal(0, 0.5)
            rest_lon = lon + np.random.normal(0, 0.5)
            
            restaurants.append({
                'business_id': f'rest_{i}_{j}',
                'name': f'Restaurant {j+1}',
                'city': city,
                'state': state,
                'latitude': rest_lat,
                'longitude': rest_lon,
                'stars': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], 
                                        p=[0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.1, 0.05]),
                'review_count': np.random.randint(5, 150),  # Reducido de 5-1000 a 5-150
                'categories': np.random.choice(['Mexican', 'Italian', 'Chinese', 'American', 'Japanese'])
            })
    
    return pd.DataFrame(restaurants), states_df

@st.cache_data
def load_sentiment_sample():
    """Generar datos de muestra para análisis de sentimientos"""
    np.random.seed(42)
    sentiments = ['Positive', 'Negative', 'Neutral']
    weights = [0.6, 0.25, 0.15]  # Más positivos que negativos
    
    data = []
    for i in range(1000):
        sentiment = np.random.choice(sentiments, p=weights)
        confidence = np.random.uniform(0.5, 0.99)
        
        data.append({
            'review_id': f'review_{i}',
            'predicted_sentiment': sentiment,
            'confidence': confidence,
            'review_stars': np.random.randint(1, 6)
        })
    
    return pd.DataFrame(data)

@st.cache_data
def load_topics_sample():
    """Generar datos de muestra para tópicos"""
    topics = [
        {'topic_id': 0, 'size': 1500, 'name': 'Service Quality', 'sentiment': 'Mixed'},
        {'topic_id': 1, 'size': 1200, 'name': 'Food Quality', 'sentiment': 'Positive'},
        {'topic_id': 2, 'size': 800, 'name': 'Pricing', 'sentiment': 'Negative'},
        {'topic_id': 3, 'size': 700, 'name': 'Ambiance', 'sentiment': 'Positive'},
        {'topic_id': 4, 'size': 600, 'name': 'Wait Times', 'sentiment': 'Negative'},
    ]
    
    return pd.DataFrame(topics)

# ==========================================
# TÍTULO PRINCIPAL
# ==========================================
st.markdown('<h1 class="main-title">🍽️ Restaurant Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Análisis de Datos y NLP para el Sector Gastronómico</p>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR MEJORADO
# ==========================================
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">Navegación</h2>', unsafe_allow_html=True)
    
    page = st.selectbox(
        "Selecciona una sección:",
        ["🏠 Dashboard Principal", "🗺️ Análisis Geográfico", "📊 Análisis Exploratorio", 
         "💭 Análisis de Sentimientos", "🎯 Modelado de Tópicos"],
        index=0
    )
    
    st.markdown("---")
    
    # Información del proyecto
    st.markdown("""
    <div class="info-box">
        <h4>📈 Proyecto TFM</h4>
        <p><strong>Tecnologías:</strong></p>
        <ul>
            <li>🤖 RoBERTa (Sentiment)</li>
            <li>🎯 BERTopic (Topics)</li>
            <li>📊 Plotly (Visualizations)</li>
            <li>🗺️ Folium (Maps)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# CARGAR DATOS
# ==========================================
with st.spinner("⏳ Cargando datos..."):
    df_restaurants, df_states = load_sample_data()
    df_sentiment = load_sentiment_sample()
    df_topics = load_topics_sample()

# Load the RoBERTa sentiment analysis model
with st.spinner("🤖 Loading RoBERTa sentiment model..."):
    sentiment_model = load_sentiment_model()

# ==========================================
# PÁGINAS DEL DASHBOARD
# ==========================================

if page == "🏠 Dashboard Principal":
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_restaurants = len(df_restaurants)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_restaurants:,}</div>
            <div class="metric-label">🏪 Restaurantes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = df_restaurants['stars'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_rating:.1f}</div>
            <div class="metric-label">⭐ Rating Promedio</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_reviews = df_restaurants['review_count'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_reviews:,}</div>
            <div class="metric-label">📝 Total Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_confidence = df_sentiment['confidence'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_confidence:.2f}</div>
            <div class="metric-label">🎯 Confianza ML</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Distribución por Estado")
        state_counts = df_restaurants['state'].value_counts()
        
        fig_states = px.bar(
            x=state_counts.values,
            y=state_counts.index,
            orientation='h',
            title="Restaurantes por Estado",
            color=state_counts.values,
            color_continuous_scale='Blues'
        )
        fig_states.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
        st.plotly_chart(fig_states, use_container_width=True)
    
    with col2:
        st.subheader("💭 Distribución de Sentimientos")
        sentiment_counts = df_sentiment['predicted_sentiment'].value_counts()
        
        colors = {'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#f59e0b'}
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Análisis de Sentimientos",
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        fig_sentiment.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

elif page == "🗺️ Análisis Geográfico":
    st.header("🗺️ Distribución Geográfica de Restaurantes")
    
    # Filtros en sidebar
    with st.sidebar:
        st.markdown("### 🔍 Filtros")
        
        selected_states = st.multiselect(
            "Estados:",
            options=df_restaurants['state'].unique(),
            default=df_restaurants['state'].unique()[:5]
        )
        
        min_rating = st.slider(
            "Rating mínimo:",
            min_value=1.0,
            max_value=5.0,
            value=1.0,
            step=0.5
        )
        
        min_reviews = st.slider(
            "Reviews mínimas:",
            min_value=0,
            max_value=500,
            value=0,
            step=25
        )
    
    # Filtrar datos
    df_filtered = df_restaurants[
        (df_restaurants['state'].isin(selected_states)) &
        (df_restaurants['stars'] >= min_rating) &
        (df_restaurants['review_count'] >= min_reviews)
    ]
    
    if not df_filtered.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Mapa Interactivo de Estados Unidos")
            
            # Crear mapa base centrado en USA
            center_lat = 39.8283
            center_lon = -98.5795
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='CartoDB dark_matter'
            )
            
            # Agregar marcadores por estado
            for state in selected_states:
                state_data = df_filtered[df_filtered['state'] == state]
                if not state_data.empty:
                    # Usar coordenadas promedio del estado
                    avg_lat = state_data['latitude'].mean()
                    avg_lon = state_data['longitude'].mean()
                    count = len(state_data)
                    avg_rating = state_data['stars'].mean()
                    
                    # Color basado en rating promedio
                    if avg_rating >= 4.0:
                        color = 'green'
                    elif avg_rating >= 3.0:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    folium.CircleMarker(
                        location=[avg_lat, avg_lon],
                        radius=count/50,  # Tamaño basado en cantidad
                        popup=f"""
                        <b>{state}</b><br>
                        Restaurantes: {count}<br>
                        Rating promedio: {avg_rating:.1f}⭐<br>
                        Reviews totales: {state_data['review_count'].sum():,}
                        """,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Mostrar mapa
            map_data = st_folium(m, width=700, height=500)
        
        with col2:
            st.subheader("📊 Estadísticas")
            
            # Métricas filtradas
            st.metric("Restaurantes mostrados", len(df_filtered))
            st.metric("Rating promedio", f"{df_filtered['stars'].mean():.1f}⭐")
            st.metric("Reviews totales", f"{df_filtered['review_count'].sum():,}")
            
            # Top ciudades
            st.markdown("### 🏙️ Top Ciudades")
            city_stats = df_filtered.groupby('city').agg({
                'business_id': 'count',
                'stars': 'mean'
            }).round(2).sort_values('business_id', ascending=False).head(5)
            
            for city, row in city_stats.iterrows():
                st.markdown(f"""
                **{city}**  
                🏪 {row['business_id']} restaurantes  
                ⭐ {row['stars']:.1f} rating
                """)
    else:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")

elif page == "📊 Análisis Exploratorio":
    st.header("📊 Análisis Exploratorio de Datos")
    
    # Gráficos lado a lado
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⭐ Distribución de Ratings")
        
        fig_ratings = px.histogram(
            df_restaurants,
            x='stars',
            nbins=10,
            title="Distribución de Calificaciones",
            color_discrete_sequence=['#4f46e5']
        )
        fig_ratings.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            xaxis_title="Rating (Estrellas)",
            yaxis_title="Número de Restaurantes"
        )
        st.plotly_chart(fig_ratings, use_container_width=True)
    
    with col2:
        st.subheader("📝 Distribución de Reviews")
        
        # Usar escala log para mejor visualización
        df_restaurants['log_reviews'] = np.log1p(df_restaurants['review_count'])
        
        fig_reviews = px.histogram(
            df_restaurants,
            x='log_reviews',
            nbins=20,
            title="Reviews por Restaurante (Log Scale)",
            color_discrete_sequence=['#10b981']
        )
        fig_reviews.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            xaxis_title="Log(Reviews + 1)",
            yaxis_title="Número de Restaurantes"
        )
        st.plotly_chart(fig_reviews, use_container_width=True)
    
    # Análisis por categorías
    st.subheader("🍽️ Análisis por Categorías")
    
    category_stats = df_restaurants.groupby('categories').agg({
        'business_id': 'count',
        'stars': 'mean',
        'review_count': 'mean'
    }).round(2)
    category_stats.columns = ['Cantidad', 'Rating Promedio', 'Reviews Promedio']
    category_stats = category_stats.sort_values('Cantidad', ascending=False)
    
    fig_categories = px.bar(
        category_stats.reset_index(),
        x='Cantidad',
        y='categories',
        orientation='h',
        title="Restaurantes por Categoría",
        color='Rating Promedio',
        color_continuous_scale='Viridis'
    )
    fig_categories.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa')
    )
    st.plotly_chart(fig_categories, use_container_width=True)
    
    # Correlación ratings vs reviews
    st.subheader("🔗 Correlación: Ratings vs Reviews")
    
    fig_scatter = px.scatter(
        df_restaurants,
        x='review_count',
        y='stars',
        color='categories',
        title="Relación entre Número de Reviews y Rating",
        trendline="ols",
        hover_data=['city', 'state']
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa')
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "💭 Análisis de Sentimientos":
    st.header("💭 Análisis de Sentimientos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🧪 Text Analyzer")
        
        # Model information
        if TRANSFORMERS_AVAILABLE and sentiment_model is not None:
            st.info("🤖 **Using RoBERTa Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` - Optimized for English text sentiment analysis")
        else:
            st.warning("⚠️ **Simulation Mode**: RoBERTa model not available. Using simulated results for demonstration.")
        
        # Entrada de texto
        user_text = st.text_area(
            "Write a review to analyze:",
            placeholder="Example: The food was delicious and the service was excellent. I will definitely return.",
            height=100
        )
        
        if st.button("🚀 Analyze Sentiment", type="primary"):
            if user_text.strip():
                # Use the actual RoBERTa model for sentiment analysis
                with st.spinner("🤖 Analyzing sentiment..."):
                    sentiment_results = analyze_sentiment_text(user_text, sentiment_model)
                
                # Process the results from the model
                sentiments = []
                probabilities = []
                
                for result in sentiment_results:
                    label = result['label'].capitalize()
                    if label == 'Negative':
                        sentiments.append('Negative')
                    elif label == 'Neutral':
                        sentiments.append('Neutral')
                    else:  # Positive
                        sentiments.append('Positive')
                    probabilities.append(result['score'])
                
                result_data = pd.DataFrame({
                    'Sentiment': sentiments,
                    'Probability': probabilities
                })
                
                # Gráfico de barras
                fig = px.bar(
                    result_data,
                    x='Sentiment',
                    y='Probability',
                    color='Probability',
                    color_continuous_scale='RdYlGn',
                    title="Sentiment Analysis"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Resultado principal
                max_sentiment = sentiments[np.argmax(probabilities)]
                max_prob = np.max(probabilities)
                
                color_map = {'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#f59e0b'}
                color = color_map[max_sentiment]
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}20, {color}10);
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    <h3 style="color: {color}; margin: 0;">
                        Sentiment: {max_sentiment}
                    </h3>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0; color: #fafafa;">
                        Confidence: {max_prob:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with col2:
        st.subheader("📋 Examples")
        
        examples = [
            "🎉 Amazing experience! The food was delicious and the service exceptional.",
            "😞 Very disappointing. The food arrived cold and the service was terrible.",
            "😐 The restaurant is okay, nothing special but not bad either.",
            "❤️ My favorite place! I always come back for the incredible pasta.",
            "💸 Too expensive for the quality they offer. Not worth it."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.text_area("Selected text:", value=example.split(maxsplit=1)[1], key="example_display", disabled=True)
    
    # Estadísticas del dataset
    st.markdown("---")
    st.subheader("📈 Estadísticas del Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_pct = (df_sentiment['predicted_sentiment'] == 'Positive').mean() * 100
        st.metric("% Sentimientos Positivos", f"{positive_pct:.1f}%")
    
    with col2:
        negative_pct = (df_sentiment['predicted_sentiment'] == 'Negative').mean() * 100
        st.metric("% Sentimientos Negativos", f"{negative_pct:.1f}%")
    
    with col3:
        high_conf = (df_sentiment['confidence'] >= 0.8).mean() * 100
        st.metric("% Alta Confianza (≥0.8)", f"{high_conf:.1f}%")

elif page == "🎯 Modelado de Tópicos":
    st.header("🎯 Modelado de Tópicos")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Tópicos Principales")
        
        # Gráfico de barras de tópicos
        fig_topics = px.bar(
            df_topics,
            x='size',
            y='name',
            orientation='h',
            title="Tamaño de Tópicos Identificados",
            color='sentiment',
            color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444', 'Mixed': '#f59e0b'}
        )
        fig_topics.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
        st.plotly_chart(fig_topics, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Distribución por Sentimiento")
        
        sentiment_by_topic = df_topics['sentiment'].value_counts()
        
        fig_sentiment_topics = px.pie(
            values=sentiment_by_topic.values,
            names=sentiment_by_topic.index,
            title="Sentimiento de Tópicos",
            color=sentiment_by_topic.index,
            color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444', 'Mixed': '#f59e0b'}
        )
        fig_sentiment_topics.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
        st.plotly_chart(fig_sentiment_topics, use_container_width=True)
    
    # Tabla detallada de tópicos
    st.subheader("📋 Detalle de Tópicos")
    
    # Formatear la tabla
    df_topics_display = df_topics.copy()
    df_topics_display['Porcentaje'] = (df_topics_display['size'] / df_topics_display['size'].sum() * 100).round(1)
    df_topics_display = df_topics_display[['name', 'size', 'Porcentaje', 'sentiment']]
    df_topics_display.columns = ['Tópico', 'Documentos', 'Porcentaje (%)', 'Sentimiento']
    
    st.dataframe(
        df_topics_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Insights principales
    st.subheader("💡 Insights Principales")
    
    insights = [
        "🔍 **Calidad del Servicio** es el tópico más discutido (Mixed sentiment)",
        "🍽️ **Calidad de la Comida** genera principalmente sentimientos positivos",
        "💰 **Precios** es una preocupación recurrente (Negative sentiment)",
        "🏠 **Ambiente** del restaurante es bien valorado por los clientes",
        "⏰ **Tiempos de Espera** representan el principal punto de mejora"
    ]
    
    for insight in insights:
        st.markdown(insight)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0a0a0; padding: 1rem 0;">
    <p>🎓 <strong>TFM - Máster en Big Data y Business Analytics</strong></p>
    <p>📊 Dashboard desarrollado con Streamlit | 🤖 Análisis con RoBERTa y BERTopic</p>
</div>
""", unsafe_allow_html=True)

# Activar el entorno virtual
# source .venv/bin/activate 

# Navegar al directorio de la aplicación
# cd app 

# Ejecutar la aplicación   
# streamlit run streamlit_app.py 

