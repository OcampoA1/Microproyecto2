# Microproyecto 2 — Clasificación de textos por ODS

Entrega reproducible para clasificar textos en español según los Objetivos de Desarrollo Sostenible mediante TF-IDF, análisis semántico latente, SVD truncado y LinearSVC calibrado.

- **Autores:** Alejandro Ocampo Rojas y Mateo Alvarez Lopera
- **Institución:** Universidad de los Andes
- **Asignatura:** Machine Learning No Supervisado

[![Abrir aplicación en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://microproyecto2-gcrwplkaefh7lt9c7sbryd.streamlit.app)

## Archivos principales

- `Microproyecto2_Clasificacion_ODS.ipynb`: análisis completo, documentado y ejecutado.
- `Microproyecto2_Clasificacion_ODS.html`: versión estática para entregar.
- `Datos_textosODS.xlsx`: conjunto de datos suministrado.
- `deploy_streamlit/`: aplicación, controlador, artefacto y dependencias para Streamlit Community Cloud.
- `requirements-notebook.txt`: dependencias necesarias para reproducir el análisis.

## Ejecución local

Se recomienda Python 3.11 o superior y un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
jupyter notebook Microproyecto2_Clasificacion_ODS.ipynb
```

El notebook usa rutas relativas; debe ejecutarse desde esta carpeta. Al ejecutarlo de principio a fin vuelve a entrenar y actualiza `deploy_streamlit/resources/models/modelo_ods.joblib`.

## Aplicación Streamlit

Después de instalar las dependencias y de ejecutar el notebook:

```bash
streamlit run deploy_streamlit/streamlit_app.py
```

La aplicación acepta texto libre, presenta el ODS predicho y muestra las tres clases con mayor probabilidad calibrada. La carpeta de despliegue sigue la separación entre interfaz, controlador y recursos propuesta en la guía del curso.

## Despliegue en Community Cloud

La aplicación pública está disponible en [microproyecto2-gcrwplkaefh7lt9c7sbryd.streamlit.app](https://microproyecto2-gcrwplkaefh7lt9c7sbryd.streamlit.app).

1. Subir el proyecto a un repositorio público de GitHub mediante Git.
2. En Streamlit Community Cloud elegir `deploy_streamlit/streamlit_app.py` como archivo principal.
3. Seleccionar Python 3.11 en la configuración avanzada.
4. Desplegar y compartir la URL `streamlit.app` generada.

El modelo pesa cerca de 66 MB, por lo que debe subirse mediante Git desde terminal o GitHub Desktop y no mediante el cargador web de GitHub.

Para la entrega obligatoria solicitada en el enunciado se adjuntan el notebook en formatos `.ipynb` y `.html`. El conjunto de datos, las dependencias y la aplicación Streamlit acompañan estos archivos como material reproducible y como evidencia de la bonificación opcional.

## Alcance

El archivo suministrado contiene únicamente etiquetas de los ODS 1 a 16. El ODS 17 no está representado y, por tanto, no es una salida posible del modelo.
