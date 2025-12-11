import streamlit as st
import torch
import faiss
import json
import numpy as np
from PIL import Image
from torchvision import transforms
import os

from Source.feature_extractor import MyVGG16, MyResNet50, MyInceptionV3, MyColorHistogram, MyLBP

st.set_page_config(page_title="Buscador de Arte CBIR", layout="wide")

BASE_IMAGE_PATH = "./Data/dataset/training_set"
METADATA_PATH = "./Data/image_metadata.json"

MODELS_CONFIG = {
    "VGG16": {
        "class": MyVGG16,
        "index_path": "./Data/feature/VGG16.index",
        "description": "Modelo clásico, bueno para texturas."
    },
    "ResNet50": {
        "class": MyResNet50, 
        "index_path": "./Data/feature/ResNet50.index",
        "description": "Modelo residual profundo."
    },
    "InceptionV3": {
        "class": MyInceptionV3,
        "index_path": "./Data/feature/InceptionV3.index",
        "description": "Modelo con arquitectura de módulos Inception."
    },
    "ColorHistogram": {
        "class": MyColorHistogram,
        "index_path": "./Data/feature/ColorHistogram.index",
        "description": "Extractor basado en histogramas de color."
    },
    "LBP": {
        "class": MyLBP,
        "index_path": "./Data/feature/LBP.index",
        "description": "Extractor basado en patrones binarios locales."
    }
}

@st.cache_resource
def load_resources(model_name):
    print(f"Cargando recursos para: {model_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODELS_CONFIG[model_name]
    
    model_class = config["class"] 
    model = model_class(device=device)
    
    if not os.path.exists(config["index_path"]):
        raise FileNotFoundError(f"No se encontró el índice en {config['index_path']}")
        
    index = faiss.read_index(config["index_path"])
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    unique_categories = sorted(list(set([m['category'] for m in metadata])))
        
    return model, index, metadata, unique_categories, device

with st.sidebar:
    st.header("Configuración")
    available_models = list(MODELS_CONFIG.keys())
    selected_model_name = st.selectbox("Selecciona el Extractor:", available_models)
    st.caption(MODELS_CONFIG[selected_model_name]["description"])
    st.divider()
    
    try:
        model, index, metadata, unique_categories, device = load_resources(selected_model_name)
        st.success(f"{selected_model_name} cargado.", icon="✅")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
        
    num_results = st.slider("Resultados a mostrar", 1, 20, 5)
    
    st.divider()
    st.subheader("Validación")
    query_true_category = st.selectbox(
        "¿Cuál es la categoría real de tu imagen?", 
        options=unique_categories
    )
    
    uploaded_file = st.file_uploader("Subir imagen query...", type=["jpg", "png", "jpeg"])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# LÓGICA DE BÚSQUEDA
def search_similar_images(query_image, k=5):
    img_tensor = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.extract_features(img_tensor)
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        features = features.astype('float32')
    
    distances, indices = index.search(features, k)
    return distances[0], indices[0]

# INTERFAZ PRINCIPAL
st.title("Sistema de Recuperación de Imágenes")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 3])
    
    image_query = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Tu Imagen")
        st.image(image_query, caption=f"Categoría definida: {query_true_category}", use_container_width=True)
    
    with col2:
        st.subheader(f"Resultados ({selected_model_name})")
        
        with st.spinner('Analizando...'):
            dists, idxs = search_similar_images(image_query, k=num_results)
            
            results_meta = []
            matches = 0
            
            for dist, idx in zip(dists, idxs):
                if idx == -1: continue
                meta = metadata[idx]
                is_match = (meta['category'] == query_true_category)
                if is_match: matches += 1
                
                results_meta.append({
                    "meta": meta,
                    "dist": dist,
                    "match": is_match
                })
            
            # 1. Cálculo del porcentaje
            total_results = len(results_meta)
            accuracy_score = (matches / total_results) * 100 if total_results > 0 else 0

            # Definimos el color y el mensaje
            msg_type = st.success if accuracy_score > 50 else st.info

            # Creamos dos columnas: una para el número grande, otra para el texto
            col_metric, col_desc = st.columns([1, 3])

            with col_metric:
            # Muestra el número en grande con una flecha verde si es alto
                st.metric(
                    label="Precisión de Categoría", 
                    value=f"{accuracy_score:.1f}%", 
                    delta=f"{matches}/{total_results} Aciertos"
                )

            with col_desc:
                st.caption(f"Categoría buscada: **{query_true_category}**")
                # Barra de progreso visual (0.0 a 1.0)
                st.progress(accuracy_score / 100)
                st.markdown(
                    "Nota: Esta puntuación corresponde a los resultados que pertenecen a la misma cateogría seleccionada por el usuario, algunas imágenes pueden pertenecer a diferentes clases pero son visualmente similares."
                )
            #  MOSTRAR IMÁGENEs
            cols = st.columns(3)
            for i, item in enumerate(results_meta):
                meta = item['meta']
                dist = item['dist']
                match = item['match']
                
                fname = meta['filename']
                category = meta['category']
                full_path = os.path.join(BASE_IMAGE_PATH, category, fname)
                
                
                border_color = "✅" if match else "⚠️"
                
                with cols[i % 3]:
                    try:
                        img_result = Image.open(full_path)
                        st.image(
                            img_result, 
                            caption=f"{border_color} {category}\nDist: {dist:.2f}", 
                            use_container_width=True
                        )
                    except FileNotFoundError:
                        st.warning(f"Falta: {fname}")

else:
    st.info("👆 Selecciona configuración y sube imagen.")