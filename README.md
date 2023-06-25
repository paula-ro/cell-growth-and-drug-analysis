# Evaluación de la dosis de un fármaco en células tumorales

Este repositorio forma parte del trabajo fin de grado de Paula Romero Jiménez (Grado Biotecnología, itinerario computacional, UPM).

A partir de datos de absorbancia obtenidos en ensayos MTT realizados a varios tiempos, se han calculado las tasas de crecimiento de las células. Se obtienen tasas de crecimiento diferentes según la concentración de fármaco con la que se ha realizado el tratamiento. Esta relación se puede ajustar mediante una exponencial decreciente, y a partir de este ajuste calcular el IC50 y otros parámetros, como la concentración a la que la tasa de crecimiento es 0.


## Contenido

- [Descripción](#descripción)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Uso](#uso)

## Descripción

En este repositorio se encuentra un cuaderno de Jupyter y un script de Python que contienen funciones útiles para el análisis de datos:

- El script **funciones.py** contiene una serie de funciones que han sido diseñadas y desarrolladas específicamente para el procesamiento y análisis de datos de absorbancia en relación con la concentración de un fármaco.

- El cuaderno de Jupyter **programa calculador.ipynb** que presenta el programa en un entorno interactivo implementando las funciones del script. El cuaderno guía paso a paso al usuario a través del flujo de trabajo, desde la lectura de los datos hasta los análisis y representaciones finales.

Además la carpeta **datos** contiene unos archivos Excel con datos de absorbancia utilizados en este trabajo.




## Estructura del Repositorio

```
├── datos
├── README.md
├── funciones.py
└── programa calculador.ipynb
```

## Requisitos

- Python 3.6 o superior.
- Jupyter Notebook instalado (opcional).


## Uso

1. Abre el cuaderno de Jupyter `notebook.ipynb` en Jupyter Notebook o JupyterLab.

2. Sigue las instrucciones y ejecuta las celdas para ver ejemplos de uso de las funciones.

3. Si deseas utilizar las funciones en otro proyecto, puedes importar el módulo `funciones.py` en tu código:

```python
from funciones import funcion1, funcion2

# Utiliza las funciones en tu código
resultado = funcion1(argumento)
```

