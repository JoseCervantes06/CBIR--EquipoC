# CBIR--EquipoC

... (Descripción y Contenido Principal) ...

## 🛠️ Entorno de ejecución 

Para clonar, instalar y ejecutar este proyecto correctamente, necesitarás las siguientes herramientas y dependencias:

### 1. Requisitos de Software

Asegúrate de tener instalados:

* **Python:** Versión 3.11
* **Gestor de paquetes:** `pip` (recomendado) o `conda`.
* **Git:** Para clonar el repositorio.

### 2. Instalar Dependencias de Python

Todas las librerías necesarias se encuentran especificadas en el archivo `requirements.txt`. Ejecuta el siguiente comando en la terminal (funciona en **Windows, macOS y Linux**):

```bash
pip install -r requirements.txt
```
## 📁 Preparación del Dataset

Para ejecutar el proyecto, es necesario descargar las imágenes y organizarlas correctamente en la estructura de carpetas.

### 1. Descarga y Configuración

I.  **Descargar:** Descarga el archivo `.zip` del dataset "Art Images" desde Kaggle haciendo clic en el siguiente enlace:
    * [🔗 Kaggle - Art Images](https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving)

II.  **Descomprimir:** Extrae el contenido del archivo `.zip`.

III.  **Organizar:** Mueve la carpeta descomprimida llamada `dataset` dentro de la carpeta `Data` de este proyecto.

### 2. Estructura del Proyecto

Para que el proyecto funcione sin errores, asegúrate de que tu directorio de trabajo tenga exactamente la siguiente estructura:

```text
CBIR--EquipoC/
│
├── Data/
│   ├── Feature/       # Carpeta para características extraídas
│   └── dataset/       # Carpeta con las imágenes 
│
├── source/
│   ├── feature_extractor.py   
│   ├── images_extractor.ipynb    
│
├── main.py            # Archivo principal de ejecución
└── requirements.txt
