# Trabajo Final de Máster - Análisis de Opiniones de Restaurantes 🍽️

Este proyecto forma parte del Trabajo Final de Máster en Big Data, enfocado en el análisis de opiniones de restaurantes utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) sobre el dataset de Yelp.

## Descripción del Proyecto

El proyecto implementa un pipeline completo de análisis de datos que incluye:

- **Ingesta y preprocesamiento de datos** del dataset de Yelp
- **Análisis exploratorio** de reseñas y datos de negocios
- **Análisis de sentimientos** utilizando modelos pre-entrenados
- **Modelado de tópicos** con BERTopic para identificar temas principales
- **Dashboard interactivo** para visualización de resultados

## 🏗️ Estructura del Proyecto

```
tfm-proyecto/
├── app/                          # Dashboard interactivo
│   ├── streamlit_app.py         # Aplicación principal
│   └── utils.py                 # Utilidades
├── notebooks/                   # Análisis en Jupyter
│   ├── data-ingestion/         # Ingesta de datos
│   ├── data-analysis/          # Análisis exploratorio
│   ├── sentimental-analysis/   # Análisis de sentimientos
│   └── topic-modeling/         # Modelado de tópicos
├── data/                       # Datos procesados
├── models/                     # Modelos entrenados
└── pyproject.toml             # Configuración de dependencias
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8+
- GPU recomendada para el modelado de tópicos (opcional)

### Instalación con UV (Recomendado)

```bash
# Instalar UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar el repositorio
git clone <repository-url>
cd tfm-proyecto

# Instalar dependencias
uv sync
```

### Instalación con pip

```bash
pip install -e .
```

## 📊 Uso del Proyecto

### 1. Ejecutar Notebooks de Análisis

Los notebooks están organizados por etapas del pipeline:

```bash
# Activar entorno virtual
uv shell

# Ejecutar Jupyter
jupyter lab
```

**Orden recomendado de ejecución:**
1. `notebooks/data-ingestion/` - Preparación de datos
2. `notebooks/data-analysis/` - Análisis exploratorio
3. `notebooks/sentimental-analysis/` - Análisis de sentimientos
4. `notebooks/topic-modeling/` - Modelado de tópicos

### 2. Dashboard Interactivo

Para visualizar los resultados en un dashboard web:

```bash
cd app
streamlit run streamlit_app.py
```

El dashboard incluye:
- Métricas generales del proyecto
- Análisis de sentimientos por categorías
- Exploración de tópicos identificados
- Insights de negocio

## 🎯 Principales Funcionalidades

### Análisis de Datos
- Exploración de 100,000+ reseñas de restaurantes
- Análisis estadístico de ratings, categorías y distribuciones
- Identificación de patrones en los datos

### Análisis de Sentimientos
- Clasificación automática usando modelos BERT
- Correlación entre sentimientos y ratings
- Análisis por categorías de restaurantes

### Modelado de Tópicos
- Identificación de 70 tópicos principales usando BERTopic
- Procesamiento de 43,993 documentos
- Correlación tópico-sentimiento
- Análisis de distribución de tópicos por sentimiento

### Resultados Clave
- **Tópicos gastronómicos identificados**: Mexicana, Pizza, Sushi, Italiana, China
- **Problema crítico detectado**: Servicio al cliente (87.3% sentimiento negativo)
- **Fortalezas**: Calidad culinaria y diversidad gastronómica

## 📈 Tecnologías Utilizadas

- **Python**: Lenguaje principal
- **Streamlit**: Dashboard interactivo
- **BERTopic**: Modelado de tópicos
- **Transformers**: Análisis de sentimientos
- **Pandas/NumPy**: Manipulación de datos
- **Jupyter**: Análisis exploratorio

## 📝 Documentación Adicional

- Notebooks incluyen documentación inline de cada proceso
- Código del dashboard documentado en `app/utils.py`

## 🔧 Configuración Avanzada

Para modificar parámetros del modelo o análisis, consultar:
- Configuración de BERTopic en `notebooks/topic-modeling/`
- Parámetros de sentimientos en `notebooks/sentimental-analysis/`
- Configuración del dashboard en `app/utils.py`

## 📊 Resultados y Outputs

El proyecto genera:
- Modelos entrenados en `models/`
- Datasets procesados en `data/`
- Análisis completos en notebooks Jupyter

## 🤝 Contribución

Este es un proyecto académico desarrollado como Trabajo Final de Máster. Para modificaciones o mejoras, seguir la estructura de notebooks existente y mantener la documentación actualizada.

## 📄 Licencia

Proyecto académico - Universidad/Institución correspondiente.
