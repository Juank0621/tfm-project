# 📊 Datos del Proyecto - Dataset de Yelp

Este directorio contiene los datos utilizados en el Trabajo Final de Máster sobre **Análisis de Opiniones de Restaurantes** utilizando técnicas de Procesamiento de Lenguaje Natural (NLP).

## ⚠️ Importante - Archivos de Gran Tamaño

Debido al tamaño considerable de los archivos del dataset de Yelp (varios GB), **los archivos originales no están incluidos en este repositorio**. Los datos deben descargarse directamente desde la fuente oficial.

## 🔗 Descarga del Dataset

### Dataset Oficial de Yelp para Investigación Académica

**Enlace de descarga:** https://business.yelp.com/data/resources/open-dataset/

**Requisitos:**
- Registro gratuito con email académico o institucional
- Aceptación del acuerdo de uso académico
- Descarga del archivo `yelp_dataset.tar` (aproximadamente 10.9 GB comprimido)

### Archivos Necesarios

Una vez descargado y extraído el archivo `yelp_dataset.tar`, necesitarás los siguientes archivos JSON:

#### Archivos Principales (ubicar en `data/raw/`)
- `yelp_academic_dataset_business.json` (~118 MB) - Información de 150,346 negocios
- `yelp_academic_dataset_review.json` (~5.34 GB) - 6,990,280 reseñas de usuarios
- `yelp_academic_dataset_user.json` (~3.36 GB) - Datos de usuarios
- `yelp_academic_dataset_checkin.json` (~287 MB) - Check-ins en negocios  
- `yelp_academic_dataset_tip.json` (~298 MB) - Tips cortos de usuarios

#### Documentación Oficial
- `Dataset_User_Agreement.pdf` - Acuerdo de uso del dataset

## 📁 Estructura de Datos

```
data/
├── raw/                           # Datos originales de Yelp
│   ├── yelp_academic_dataset_business.json
│   ├── yelp_academic_dataset_review.json  
│   ├── yelp_academic_dataset_user.json
│   ├── yelp_academic_dataset_checkin.json
│   ├── yelp_academic_dataset_tip.json
│   ├── yelp_dataset.tar
│   └── Dataset_User_Agreement.pdf
├── processed/                     # Datos procesados y filtrados
│   ├── businesses.json           # Negocios filtrados  
│   ├── restaurants_with_reviews.json  # Restaurantes con reseñas
│   └── reviews.json              # Reseñas procesadas
├── analysis_results/              # Resultados de análisis
│   ├── complete_analysis_summary.json
│   ├── high_quality_restaurants.json
│   ├── premium_restaurants.json
│   └── restaurants_pa.json
├── results/                       # Resultados de modelos
│   ├── sentiment_analysis.csv
│   ├── sentiment_analysis.json
│   ├── topic_assignments.csv
│   ├── topic_model_info.csv
│   ├── topic_modeling_summary.csv
│   └── topic_sentiment_analysis.csv
└── geo/                          # Datos geográficos
    └── us-states.json
```

## 🎯 Datos del Proyecto

### Volumen de Datos Procesados
- **Negocios totales:** 150,346
- **Reseñas totales:** 6,990,280  
- **Restaurantes filtrados:** 52,268
- **Reseñas de restaurantes:** 4,724,471
- **Reseñas analizadas (alta confianza):** 84,788

### Criterios de Filtrado
1. **Negocios:** Categorías relacionadas con "Restaurant" o gastronomía
2. **Reseñas:** Texto en inglés con longitud mínima de 10 caracteres
3. **Calidad:** Filtro por confianza del modelo ≥ 0.7 para análisis final

## 🚀 Instrucciones de Configuración

### 1. Descargar el Dataset
1. Visita https://business.yelp.com/data/resources/open-dataset/
2. Completa el registro académico
3. Descarga `yelp_dataset.tar`
4. Extrae los archivos en `data/raw/`

### 2. Ejecutar Pipeline de Datos
Una vez descargados los datos, ejecuta los notebooks en orden:

```bash
# 1. Ingesta y filtrado inicial
jupyter lab notebooks/data-ingestion/data_ingestion.ipynb

# 2. Análisis exploratorio  
jupyter lab notebooks/data-analysis/

# 3. Procesamiento de NLP
jupyter lab notebooks/sentimental-analysis/
jupyter lab notebooks/topic-modeling/
```

## 📋 Formato de Datos

### Business JSON Schema
```json
{
  "business_id": "string",
  "name": "string", 
  "address": "string",
  "city": "string",
  "state": "string",
  "postal_code": "string",
  "latitude": "float",
  "longitude": "float", 
  "stars": "float",
  "review_count": "int",
  "is_open": "int",
  "attributes": "object",
  "categories": "string",
  "hours": "object"
}
```

### Review JSON Schema  
```json
{
  "review_id": "string",
  "user_id": "string",
  "business_id": "string", 
  "stars": "int",
  "useful": "int",
  "funny": "int", 
  "cool": "int",
  "text": "string",
  "date": "string"
}
```

## 📝 Notas sobre Uso Académico

- ✅ **Permitido:** Investigación académica, análisis de datos, desarrollo de modelos ML/NLP
- ❌ **Prohibido:** Uso comercial, redistribución, scraping adicional
- 📄 **Referencia:** Siempre citar el dataset oficial de Yelp en publicaciones

### Cita Recomendada
```
Yelp Open Dataset. (2024). Retrieved from https://business.yelp.com/data/resources/open-dataset/
```

---

**📧 Para dudas sobre el dataset:** Consultar documentación oficial en https://business.yelp.com/data/resources/open-dataset/
