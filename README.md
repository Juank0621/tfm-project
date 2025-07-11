# 🍽️ Trabajo Final de Máster - Análisis de Opiniones de Restaurantes

Este proyecto forma parte del Trabajo Final de Máster en Big Data, enfocado en el análisis de opiniones de restaurantes utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) sobre el dataset de Yelp. El sistema procesa 150,346 negocios y 6,990,280 reseñas totales, extrayendo 4,724,471 reseñas específicas de restaurantes para análisis detallado.

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
├── app/                        # Dashboard interactivo
│   └── streamlit_app.py        # Aplicación principal
├── notebooks/                  # Análisis en Jupyter
│   ├── data-ingestion/         # Ingesta de datos
│   ├── data-analysis/          # Análisis exploratorio
│   ├── sentimental-analysis/   # Análisis de sentimientos
│   └── topic-modeling/         # Modelado de tópicos
├── tfm/                        # Documentación académica
│   ├── tfm.tex                 # Documento LaTeX principal
│   └── figures/                # Figuras e imágenes del TFM
├── data/                       # Datos procesados
├── models/                     # Modelos entrenados
├── figures/                    # Gráficos y visualizaciones
└── pyproject.toml              # Configuración de dependencias
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
- Exploración de 4,724,471 reseñas de restaurantes
- Análisis estadístico de 52,268 establecimientos gastronómicos
- Pipeline de ingesta por lotes de 5,000 documentos
- Filtrado por calidad de datos (confianza ≥ 0.7)

### Análisis de Sentimientos
- Clasificación automática usando RoBERTa pre-entrenado
- Precisión del 91.3% en correlación sentimientos-ratings
- Procesamiento de 84,788 reseñas de alta confianza
- Análisis por categorías de restaurantes y distribuciones

### Modelado de Tópicos
- Identificación de 70 tópicos principales usando BERTopic
- Procesamiento de 43,993 documentos filtrados
- Análisis de correlación tópico-sentimiento
- Distribución detallada de tópicos por categorías gastronómicas

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
- **LaTeX**: Documentación académica

## 📝 Documentación Académica

### Trabajo Final de Máster (TFM)

El proyecto incluye documentación académica completa en formato LaTeX:

```bash
# Compilar el documento académico
cd tfm
pdflatex tfm.tex
```

El documento incluye:
- Abstract en español e inglés
- Marco teórico y metodología
- Análisis detallado de resultados
- Capturas del dashboard de Streamlit
- Referencias bibliográficas

### Documentación Adicional

- Notebooks incluyen documentación inline de cada proceso
- Figuras y gráficos en `figures/` y `tfm/figures/`
- Código del dashboard autodocumentado en `app/streamlit_app.py`

## 🔧 Configuración Avanzada

Para modificar parámetros del modelo o análisis, consultar:
- Configuración de BERTopic en `notebooks/topic-modeling/`
- Parámetros de sentimientos en `notebooks/sentimental-analysis/`
- Configuración del dashboard en `app/streamlit_app.py`

## 📊 Resultados y Outputs

El proyecto genera:
- **Documento académico completo** (`tfm/tfm.pdf`) con 105 páginas
- **Modelos entrenados** en `models/`
- **Datasets procesados** en `data/`
- **Análisis completos** en notebooks Jupyter
- **Visualizaciones** en `figures/` (40+ gráficos)
- **Dashboard interactivo** con métricas en tiempo real

### Métricas Principales Logradas
- **Precisión de sentimientos**: 91.3%
- **Tópicos identificados**: 70 categorías distintas
- **Documentos procesados**: 43,993 reseñas filtradas
- **Integridad de datos**: >99% después de limpieza

## 🏆 Características Técnicas Destacadas

- **Pipeline escalable** con procesamiento por lotes de 5,000 documentos
- **Filtrado inteligente** por confianza del modelo (≥ 0.7)
- **Arquitectura modular** que permite extensiones futuras
- **Manejo robusto de errores** JSON malformados
- **Validación automática** de campos críticos
- **Dashboard responsivo** con visualizaciones interactivas

## 🤝 Contribución

Este es un proyecto académico desarrollado como Trabajo Final de Máster. Para modificaciones o mejoras, seguir la estructura de notebooks existente y mantener la documentación actualizada.

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
