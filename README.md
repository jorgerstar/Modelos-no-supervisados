# Modelos de Aprendizaje No Supervisado — Salud Mental en Adolescentes
### Semana 3 · Lab 2 | Aprendizaje Automático

---

## Descripción general

Este proyecto implementa y analiza modelos de **aprendizaje no supervisado** aplicados al dataset *Teen Mental Health Dataset* de Kaggle. El objetivo es segmentar perfiles de adolescentes según sus hábitos digitales y estado de salud mental, utilizando K-Means, DBSCAN, PCA y t-SNE. Los resultados permiten identificar grupos en riesgo y comprender patrones de comportamiento digital en jóvenes de 13 a 19 años.

> **Contexto académico:** Tarea grupal — Semana 3, Modelos No Supervisados.  
> El notebook fue desarrollado originalmente con el dataset *Mall Customers* y migrado al dataset de salud mental adolescente según indicación del docente.

---

## Estructura del repositorio

```
proyecto/
│
├── 📓 AA_Modelos_NO_Supervisados_S3_Grupo2_V2.ipynb   ← Notebook principal
├── 📄 Teen_Mental_Health_Dataset.csv                          ← Dataset
└── 📄 README.md                                               ← Este archivo
```

> **Importante:** El archivo CSV debe estar en la **misma carpeta** que el notebook para que la celda de carga funcione correctamente.

---

## Dataset

| Campo | Detalle |
|---|---|
| **Nombre** | Teen Mental Health Dataset |
| **Fuente** | [Kaggle — algozee/teenager-menthal-healy](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy) |
| **Filas** | 1,200 registros |
| **Columnas** | 13 variables |
| **Valores nulos** | Ninguno |
| **Duplicados** | Ninguno |

### Variables del dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `age` | Numérica | Edad del adolescente (13–19 años) |
| `gender` | Categórica | Género (`male` / `female`) |
| `daily_social_media_hours` | Numérica | Horas diarias en redes sociales (1–8h) |
| `platform_usage` | Categórica | Plataforma principal (`Instagram` / `TikTok` / `Both`) |
| `sleep_hours` | Numérica | Horas de sueño por noche |
| `screen_time_before_sleep` | Numérica | Horas de pantalla antes de dormir |
| `academic_performance` | Numérica | Rendimiento académico (escala numérica) |
| `physical_activity` | Numérica | Nivel de actividad física |
| `social_interaction_level` | Categórica | Nivel de interacción social (`low` / `medium` / `high`) |
| `stress_level` | Numérica | Nivel de estrés (escala 1–10) |
| `anxiety_level` | Numérica | Nivel de ansiedad (escala 1–10) |
| `addiction_level` | Numérica | Nivel de adicción a redes sociales (escala 1–10) |
| `depression_label` | Binaria | Indicador de depresión (`0` = no, `1` = sí) — **no usada como input del modelo** |

---

## Requisitos del entorno

### Entorno recomendado
- **Google Colab** (recomendado — sin instalación local requerida)
- Python 3.8 o superior

### Librerías necesarias

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Instalación

En la **Parte 1** del notebook se incluye la celda de instalación. En Google Colab, descoméntala y ejecútala:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```

> En Colab, `pandas`, `numpy`, `matplotlib` y `seaborn` ya vienen preinstalados. Ejecutar el pip install de todas formas no causa errores — simplemente confirma que ya están disponibles.

### Versiones utilizadas durante el desarrollo

| Librería | Versión mínima recomendada |
|---|---|
| Python | 3.8+ |
| pandas | 1.3+ |
| numpy | 1.21+ |
| matplotlib | 3.4+ |
| seaborn | 0.11+ |
| scikit-learn | 1.0+ |

---

## Instrucciones de ejecución

### En Google Colab

1. Abre [Google Colab](https://colab.research.google.com/)
2. Sube el archivo `.ipynb` desde **Archivo → Subir cuaderno**
3. Sube el archivo `Teen_Mental_Health_Dataset.csv` al entorno de Colab:
   - En el panel izquierdo, haz clic en el ícono de carpeta 📁
   - Haz clic en el ícono de subida y selecciona el CSV
4. Descomenta la celda `!pip install ...` en la Parte 1
5. Ejecuta todas las celdas en orden con **Runtime → Run all** o `Ctrl + F9`

### En Jupyter Notebook (local)

1. Clona el repositorio o descarga los archivos
2. Abre una terminal en la carpeta del proyecto
3. Ejecuta: `jupyter notebook`
4. Abre el archivo `.ipynb` desde el navegador
5. Ejecuta todas las celdas en orden

> Asegúrate de que `Teen_Mental_Health_Dataset.csv` esté en el **mismo directorio** que el notebook antes de ejecutar.

---

## Contenido del notebook

El notebook está organizado en 8 partes:

### Parte 1 — Preparación del entorno
Instalación y carga de todas las librerías. Configuración visual de gráficas con `seaborn` y `matplotlib`.

### Parte 2 — Análisis Exploratorio (EDA)
- Vista previa del dataset (`head`, `info`, `describe`)
- Verificación de nulos y duplicados
- Histogramas de distribución de variables numéricas
- Gráficos de barras de variables categóricas
- **Matriz de correlación** con heatmap
- Boxplots comparativos por género

### Parte 3 — Preprocesamiento
- Selección de las **8 features numéricas** relevantes para clustering
- Justificación de exclusión de variables (`age`, categóricas, `depression_label`)
- Escalado con `StandardScaler` (media 0, desviación estándar 1)
- Verificación del resultado del escalado

### Parte 4.1 — K-Means Clustering
- **Método del Codo** para determinar K óptimo (rango 1–10)
- **Silhouette Score** para validación cuantitativa (rango 2–9)
- Ajuste del modelo final con `k=4`
- Visualización de clusters en espacio de features (2 gráficos comparativos)
- **Tabla resumen** de características medias por cluster
- Interpretación de los 4 perfiles identificados

### Parte 4.2 — DBSCAN Clustering
- **Gráfica K-Distance** para selección de `eps`
- Ajuste del modelo con `eps=1.5`, `min_samples=10`
- Visualización de clusters y puntos de ruido
- Evaluación con Silhouette Score (excluyendo ruido)
- Reflexión académica sobre las limitaciones de DBSCAN en datos de comportamiento humano

### Parte 5.1 — PCA (Reducción Lineal)
- Reducción a 2 componentes principales
- Análisis de **varianza explicada** por componente y acumulada
- Análisis de **loadings** (contribución de cada variable)
- Visualización 2D coloreada por clusters K-Means y DBSCAN

### Parte 5.2 — t-SNE (Reducción No Lineal)
- Reducción a 2 dimensiones con `perplexity=40`, `init='pca'`
- Visualización coloreada por: clusters K-Means, nivel de estrés, y etiqueta de depresión (validación externa)
- Comparación visual directa **PCA vs t-SNE**

### Parte 7 — Detección de Anomalías
- **Isolation Forest** (`contamination=5%`) — identificación de perfiles extremos
- **Local Outlier Factor — LOF** (`n_neighbors=20`, `contamination=5%`)
- Comparación entre ambos métodos
- Perfil de adolescentes detectados por consenso (ambos métodos)

### Parte 8 — Análisis Comparativo y Conclusiones
- Resumen comparativo de todos los modelos con métricas
- Tabla final de perfiles de segmentación
- Reflexión sobre perfiles, diferencias entre modelos y limitaciones encontradas

---

## 🧩 Modelos implementados

| Modelo | Librería | Parámetros clave | Propósito |
|---|---|---|---|
| **K-Means** | `sklearn.cluster.KMeans` | `n_clusters=4`, `random_state=42`, `n_init=10` | Segmentación principal |
| **DBSCAN** | `sklearn.cluster.DBSCAN` | `eps=1.5`, `min_samples=10` | Clustering por densidad + detección de outliers |
| **PCA** | `sklearn.decomposition.PCA` | `n_components=2` | Reducción lineal + visualización |
| **t-SNE** | `sklearn.manifold.TSNE` | `perplexity=40`, `init='pca'`, `random_state=42` | Reducción no lineal + visualización |
| **Isolation Forest** | `sklearn.ensemble.IsolationForest` | `contamination=0.05`, `random_state=42` | Detección de anomalías |
| **LOF** | `sklearn.neighbors.LocalOutlierFactor` | `n_neighbors=20`, `contamination=0.05` | Detección de anomalías local |

---

## Resultados principales

### Segmentación K-Means (k=4)

| Cluster | Perfil | Horas redes | Estrés | Ansiedad | Adicción |
|---|---|---|---|---|---|
| 0 | **Equilibrado / Saludable** | ≈3.4h | Bajo | Bajo | Bajo |
| 1 | **Riesgo Psicológico** | ≈3.4h | Medio | **Alto (≈7.7)** | Alto |
| 2 | **Sobreexpuesto Digital** | **≈6.2h** | **Alto** | Alto | Medio |
| 3 | **Adicción Digital** | ≈5.3h | Medio | Medio | **Muy alto (≈7.9)** |

### Métricas de evaluación

| Modelo | Métrica | Valor |
|---|---|---|
| K-Means (k=4) | Silhouette Score | ≈0.09 |
| DBSCAN | Puntos de ruido | ≈97% del dataset |
| PCA | Varianza conservada (2D) | Ver notebook |
| Isolation Forest | Anomalías detectadas | 60 (5%) |
| LOF | Anomalías detectadas | 60 (5%) |

> **Nota sobre el Silhouette Score:** Un score de 0.09 es esperado en datos de comportamiento humano. Los perfiles de personas no tienen fronteras perfectamente nítidas. Esto no invalida la segmentación — la validez se complementa con interpretabilidad y análisis de medias por cluster.

---

## Limitaciones conocidas

1. **Silhouette Score bajo:** Los comportamientos humanos son continuos, no discretos. Los clusters son "difusos" por naturaleza. Se recomienda complementar con validación de expertos en salud mental.

2. **DBSCAN con alta proporción de ruido:** La distribución uniforme del dataset hace que DBSCAN no sea el algoritmo más adecuado aquí. Alternativa futura: **HDBSCAN** (variante jerárquica más robusta para distribuciones variables).

3. **Variables categóricas no incluidas en el modelo:** `gender`, `platform_usage` y `social_interaction_level` podrían enriquecer los perfiles con One-Hot Encoding o embeddings. Se excluyeron para mantener la simplicidad del modelo base.

4. **Posible dataset semi-sintético:** La distribución muy uniforme de algunas variables sugiere que los datos podrían ser generados o balanceados artificialmente. En datos reales los patrones probablemente serían más pronunciados.

5. **`depression_label` no usada como input:** Esta variable binaria se reservó intencionalmente para validación externa. En un análisis supervisado complementario podría usarse como variable objetivo.

---

## Referencias

- Dataset: [Teen Mental Health Dataset — Kaggle](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy)
- Documentación scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
- K-Means: [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- DBSCAN: [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- PCA: [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- t-SNE: [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
