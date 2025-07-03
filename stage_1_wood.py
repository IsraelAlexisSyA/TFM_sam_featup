from PIL import Image # Explicitly import Image
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision.transforms as T
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import time # Para medir el tiempo

# Importar las utilidades de FeatUp para normalización/desnormalización
from featup.util import norm, unnorm
from featup.plotting import plot_feats # Asegúrate de que esta importación sea correcta
from anomalib.models.components.sampling import KCenterGreedy # Make sure KCenterGreedy uses 'embedding' and 'sampling_ratio' as args

import sys

print("FeatUp Stage 1: Importaciones completadas.")

# --- Configuración Inicial ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224 # Tamaño de entrada para DINOv2
BACKBONE_PATCH_SIZE = 14 # Tamaño de parche para DINOv2 ViT-S/14
use_norm = True # Coherente con tu enfoque

# Las dimensiones espaciales de los mapas de características de baja resolución (H', W')
H_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 14 = 16
W_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 16 = 16
PATCHES_PER_IMAGE = H_prime * W_prime # Número de parches por mapa de características de imagen (16*16 = 256)

# Dimensiones objetivo para las máscaras después de escalar (8H', 8W')
TARGET_MASK_H = 8 * H_prime # 8 * 16 = 128
TARGET_MASK_W = 8 * W_prime # 8 * 16 = 128

# Directorio de imágenes (ajusta según tu estructura)
directorio_imagenes = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/wood/train/good'
directorio_coreset = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/wood/train/good/archivos_coreset'
archivos_imagen = sorted(glob.glob(os.path.join(directorio_imagenes, '*.png'))) # Ordenar para consistencia de índices

# --- Parámetro para Coreset Subsampling ---
CORESET_SAMPLING_RATIO = 0.1

# --- Transformación Única para todas las imágenes ---
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(), # Escala píxeles a [0, 1] y cambia a (C, H, W)
    norm # Aplica normalización por media/std (normalización ImageNet)
])

# --- Nombres de archivo para guardar/cargar ---
os.makedirs(directorio_coreset, exist_ok=True) # Ensure the coreset directory exists
coreset_features_file = os.path.join(directorio_coreset, 'coreset_features.pt')
full_bank_filenames_file = os.path.join(directorio_coreset, 'full_bank_filenames.npy')
coreset_indices_file = os.path.join(directorio_coreset, 'coreset_indices.pt')
template_features_bank_coreset_file = os.path.join(directorio_coreset, 'template_features_bank_coreset.pt')
banco_de_caracteristicas_lr_file = os.path.join(directorio_coreset, 'banco_de_caracteristicas_lr.pt')
core_bank_features_file = os.path.join(directorio_coreset, 'core_bank_features.pt')
core_bank_filenames_file = os.path.join(directorio_coreset, 'core_bank_filenames.pt')
coreset_relevant_flat_features_bank_file = os.path.join(directorio_coreset, 'coreset_relevant_flat_features_bank.pt')


# --- Define extract_dinov2_features_lr function FIRST ---
def extract_dinov2_features_lr(image_path, model, image_transform, device):
    """Extrae características de baja resolución de DINOv2 usando la transformación dada."""
    # This try-except is crucial for handling corrupted images
    try:
        input_tensor = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(input_tensor)
        return features.cpu() # Keep on CPU for concatenation
    except OSError as e:
        print(f"ERROR: No se pudo cargar o procesar la imagen '{image_path}'. Error: {e}")
        return None # Return None if image loading fails

# --- 1. Cargar modelo DINOv2 (a través de FeatUp) ---
print("Cargando modelo DINOv2...")
# Note: The 'El Coreset de características no existe. Generándolo...' message is now here.
# You might want to move it to before the Coreset generation if you always generate it.
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
dinov2_model = upsampler.model # Obtiene el modelo base DINOv2 del upsampler
dinov2_model.eval() # Pone el modelo en modo de evaluación
print("Modelo DINOv2 cargado.")

# Banco de características B = {B1, ..., Bz}
banco_de_caracteristicas_lr = []
nombres_archivos_banco = [] # Initialize list for filenames

print("Extrayendo características de baja resolución para el banco de imágenes de referencia...")
for i, ruta_imagen in enumerate(archivos_imagen):
    nombre_archivo = os.path.basename(ruta_imagen)
    
    # Use the defined function with its internal error handling
    features = extract_dinov2_features_lr(ruta_imagen, dinov2_model, transform, device)
    
    if features is not None: # Only proceed if features were successfully extracted
        banco_de_caracteristicas_lr.append(features)
        nombres_archivos_banco.append(nombre_archivo) # Guarda el nombre del archivo

print(f"Número de elementos en el banco de características (después de filtrar errores): {len(banco_de_caracteristicas_lr)}")

if banco_de_caracteristicas_lr:
    print(f"Dimensiones de características de referencia (ej. B[0]): {banco_de_caracteristicas_lr[0].shape}")
    print(f"Guardando banco de características de baja resolución en: {banco_de_caracteristicas_lr_file}")
    torch.save(banco_de_caracteristicas_lr, banco_de_caracteristicas_lr_file)
else:
    print("Advertencia: No se extrajo ninguna característica de baja resolución. El banco está vacío.")

# --- Coreset Subsampling Implementation ---
if banco_de_caracteristicas_lr: # Only proceed if there are features to sample from
    print("\nPreparando banco de características para Coreset subsampling...")
    all_lr_features_flat_list = []

    for features in banco_de_caracteristicas_lr:
        # features has shape (1, C, H', W')
        features_squeezed = features.squeeze(0) # -> (C, H', W')
        # Permute to (H', W', C) then reshape to (H'*W', C)
        features_flat = features_squeezed.permute(1, 2, 0).reshape(-1, features_squeezed.shape[0])
        all_lr_features_flat_list.append(features_flat)

    full_feature_bank_flat = torch.cat(all_lr_features_flat_list, dim=0).to(device) # Move to GPU
    print(f"Dimensión del banco de características completo (aplanado): {full_feature_bank_flat.shape}")

    # 4. Aplicar Coreset Subsampling
    print(f"Aplicando Coreset subsampling con ratio: {CORESET_SAMPLING_RATIO}...")
    start_time_coreset = time.time()

    # KCenterGreedy from Anomalib expects 'embedding' for features
    sampler = KCenterGreedy(embedding=full_feature_bank_flat, sampling_ratio=CORESET_SAMPLING_RATIO)
    coreset_indices = sampler.select_coreset_idxs()
    coreset_features = full_feature_bank_flat[coreset_indices]
    end_time_coreset = time.time()
    print(f"Tiempo para aplicar Coreset: {end_time_coreset - start_time_coreset:.4f} segundos")
    print(f"Dimensión del Coreset de características (parches aplanados): {coreset_features.shape}")

    # --- Guardar el Coreset de parches aplanados y los nombres de archivo ---
    print(f"Guardando nombres de archivo del banco completo en: {full_bank_filenames_file}")
    np.save(full_bank_filenames_file, np.array(nombres_archivos_banco))
    print(f"Guardando índices del Coreset en: {coreset_indices_file}")
    torch.save(coreset_indices, coreset_indices_file)
    print(f"Guardando el banco de características del coreset (parches aplanados) en: {template_features_bank_coreset_file}")
    torch.save(coreset_features, template_features_bank_coreset_file)

    # --- Identificar y guardar los mapas de características completos relevantes para el Coreset ---
    coreset_relevant_full_feature_maps = []
    coreset_relevant_filenames = []

    print("\nIdentificando y guardando mapas de características completos relevantes para el Coreset...")
    
    # Asegurarse de que coreset_indices sea un array NumPy
    if isinstance(coreset_indices, torch.Tensor):
        coreset_indices_np = coreset_indices.cpu().numpy()
    elif isinstance(coreset_indices, np.ndarray):
        coreset_indices_np = coreset_indices
    elif isinstance(coreset_indices, list): # In case it's unexpectedly a list
        coreset_indices_np = np.array(coreset_indices)
    else:
        raise TypeError(f"Tipo inesperado para coreset_indices: {type(coreset_indices)}")

    # Calcular los índices de las imágenes originales a partir de los coreset_indices
    original_image_indices_from_coreset = (coreset_indices_np // PATCHES_PER_IMAGE).astype(int)

    # Obtener los índices únicos de las imágenes que contribuyeron al coreset
    unique_original_image_indices = np.unique(original_image_indices_from_coreset)
    print(f"Número de imágenes originales que contribuyeron al coreset: {len(unique_original_image_indices)}")

    # Filtrar el banco de características completo para obtener solo los mapas relevantes
    for idx in unique_original_image_indices:
        if 0 <= idx < len(banco_de_caracteristicas_lr):
            coreset_relevant_full_feature_maps.append(banco_de_caracteristicas_lr[idx])
            if nombres_archivos_banco and 0 <= idx < len(nombres_archivos_banco):
                coreset_relevant_filenames.append(nombres_archivos_banco[idx])
        else:
            print(f"Advertencia: Índice de imagen original {idx} fuera de rango para banco_de_caracteristicas_lr.")

    print(f"Creada lista 'coreset_relevant_full_feature_maps' con {len(coreset_relevant_full_feature_maps)} mapas.")
    if coreset_relevant_full_feature_maps:
        print(f"Dimensiones del primer mapa relevante: {coreset_relevant_full_feature_maps[0].shape}")
    if coreset_relevant_filenames:
        print(f"Ejemplo de nombre de archivo relevante: {coreset_relevant_filenames[0]}")

    # --- Guardar la lista de mapas de características completos relevantes para el Coreset ---
    print(f"Guardando los mapas de características completos relevantes del coreset en: {core_bank_features_file}")
    torch.save(coreset_relevant_full_feature_maps, core_bank_features_file)
    # --- Guardar los nombres de archivo de los mapas relevantes del Coreset ---
    print(f"Guardando los nombres de archivo relevantes del coreset en: {core_bank_filenames_file}")
    torch.save(coreset_relevant_filenames, core_bank_filenames_file)

    # --- NUEVO: Preparar y guardar el banco de características del coreset relevante, ya aplanado y apilado ---
    print("\nPreparando el banco de características del coreset relevante (aplanado y apilado) para KNN...")
    start_time_flatten_stack = time.time()
    coreset_relevant_flat_features_list = []
    
    # Make sure to handle the case where feature_map is a PyTorch tensor (which it should be)
    for feature_map in coreset_relevant_full_feature_maps:
        # feature_map has shape (1, C, H', W'). Flatten to (C*H'*W')
        coreset_relevant_flat_features_list.append(feature_map.flatten().cpu()) # Ensure it's on CPU before numpy conversion

    # Stack the list of tensors into a single tensor
    if coreset_relevant_flat_features_list:
        coreset_relevant_flat_features_bank = torch.stack(coreset_relevant_flat_features_list).to(device)
    else:
        # Handle case where list is empty
        coreset_relevant_flat_features_bank = torch.empty(0).to(device) # Or appropriate empty tensor

    end_time_flatten_stack = time.time()
    print(f"Tiempo para aplanar y apilar el banco relevante del coreset: {end_time_flatten_stack - start_time_flatten_stack:.4f} segundos")
    print(f"Dimensión del banco de características del coreset relevante (aplanado y apilado): {coreset_relevant_flat_features_bank.shape}")

    print(f"Guardando el banco de características del coreset relevante (aplanado y apilado) en: {coreset_relevant_flat_features_bank_file}")
    torch.save(coreset_relevant_flat_features_bank, coreset_relevant_flat_features_bank_file)

else:
    print("No se generó el coreset ni los mapas de características relevantes debido a la falta de características extraídas.")

print("\nFeatUp Stage 1 completado. Todos los archivos necesarios han sido generados y guardados.")
print(f"Número total de índices en el Coreset: {len(coreset_indices) if coreset_indices is not None else 0}")

print("\n### Resumen de archivos guardados ###")
print(f"  - Coreset de parches aplanados para KNN (shape: N, C): {template_features_bank_coreset_file}")
print(f"  - Banco de características completos de baja resolución (lista de tensores (1, C, H', W')): {banco_de_caracteristicas_lr_file}")
print(f"  - Banco de características completos relevantes del Coreset (lista de tensores (1, C, H', W')): {core_bank_features_file}")
print(f"  - Banco de características relevantes del Coreset, aplanado para HR (shape: N, C*H'*W'): {coreset_relevant_flat_features_bank_file}")
print(f"  - Nombres de archivo del banco completo: {full_bank_filenames_file}")
print(f"  - Nombres de archivo de los mapas relevantes del Coreset: {core_bank_filenames_file}")
print(f"  - Índices del Coreset: {coreset_indices_file}")