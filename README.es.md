# Tutorial de una aplicacion web de ML con Flask

![Banner del proyecto](reports/figures/app_banner.png)

**Idioma / Language:** [English](README.md) | Español

Este proyecto entrena un modelo de Machine Learning con el **Iris Dataset** de UCI y lo integra en una aplicacion web construida con **Flask**. La interfaz permite introducir medidas de una flor y obtener la especie estimada: `setosa`, `versicolor` o `virginica`.

## Objetivo

- Buscar y comprender un dataset simple.
- Entrenar y optimizar un modelo de Machine Learning.
- Analizar resultados y variables importantes.
- Crear una aplicacion web con Flask para usar el modelo.
- Preparar el proyecto para despliegue en Render.

## Dataset

Se utiliza el **UCI Iris Dataset**, disponible a traves de `sklearn.datasets.load_iris`.

Variables:

- `sepal_length_cm`
- `sepal_width_cm`
- `petal_length_cm`
- `petal_width_cm`

Target:

- `species`

## Modelo

El pipeline usa:

- `StandardScaler`
- `RandomForestClassifier`
- `GridSearchCV` para optimizar hiperparametros

Metricas y resultados se guardan en:

- `models/iris_metrics.json`
- `reports/figures/`

## Aplicacion Flask

La app principal esta en:

```text
app.py
```

Rutas:

- `/` formulario de prediccion.
- `/health` verificacion rapida del servicio.

## Estructura

```text
.
├── app.py
├── data/
│   ├── raw/iris.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── iris_classifier.joblib
│   └── iris_metrics.json
├── reports/figures/
│   ├── app_banner.png
│   ├── feature_importance.png
│   ├── petal_scatter.png
│   └── species_distribution.png
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
├── static/styles.css
├── templates/index.html
├── Procfile
├── render.yaml
└── requirements.txt
```

## Ejecutar localmente

```bash
pip install -r requirements.txt
python src/app.py
python app.py
```

Luego abre:

```text
http://localhost:5000
```

## Render

El repositorio incluye `Procfile` y `render.yaml`.

Pasos:

1. Crear un nuevo Web Service en Render.
2. Conectar este repositorio de GitHub.
3. Usar:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Copiar aqui la URL publica cuando Render la genere.

URL de Render:

```text
Pendiente de pegar despues de crear el servicio en Render.
```

## Recursos externos

- UCI Iris Dataset via scikit-learn.
- Flask documentation.
- Render Web Services documentation.
