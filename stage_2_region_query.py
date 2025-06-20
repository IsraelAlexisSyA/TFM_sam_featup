import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

# FeatUp utilities
from featup.util import norm, unnorm
from featup.plotting import plot_feats

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import median_abs_deviation

# Anomaly region detection and visualization
from skimage import measure
import matplotlib.patches as patches

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2

# PCA for manual visualization
from sklearn.decomposition import PCA

# --- Configuración ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224 # DINOv2 input size
BACKBONE_PATCH_SIZE = 14 # DINOv2 ViT-S/14 patch size
use_norm = True

H_prime = input_size // BACKBONE_PATCH_SIZE
W_prime = input_size // BACKBONE_PATCH_SIZE

# Directorios
TRAIN_GOOD_DIR = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/train/good'
PLOT_SAVE_ROOT_DIR = '/home/imercatoma/FeatUp/plots_single/cut'
# --- Imagen de Consulta ---
query_image_path = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/test/cut/003.png'
os.makedirs(PLOT_SAVE_ROOT_DIR, exist_ok=True)

HEATMAPS_SAVE_DIR = os.path.join(PLOT_SAVE_ROOT_DIR, 'individual_heatmaps')
os.makedirs(HEATMAPS_SAVE_DIR, exist_ok=True)

ANOMALY_REGIONS_SAVE_DIR = os.path.join(PLOT_SAVE_ROOT_DIR, 'detected_anomaly_regions')
os.makedirs(ANOMALY_REGIONS_SAVE_DIR, exist_ok=True)

FEATUP_PLOTS_DIR = os.path.join(PLOT_SAVE_ROOT_DIR, 'featup_feature_plots')
os.makedirs(FEATUP_PLOTS_DIR, exist_ok=True)

# Coreset file paths
core_bank_filenames_file = os.path.join(TRAIN_GOOD_DIR, 'core_bank_filenames.pt')
coreset_relevant_flat_features_bank_file = os.path.join(TRAIN_GOOD_DIR, 'coreset_relevant_flat_features_bank.pt')
template_features_bank_coreset_file = os.path.join(TRAIN_GOOD_DIR, 'template_features_bank_coreset.pt')

# --- Cargar Datos del Coreset ---
print("Cargando datos del coreset...")
try:
    coreset_relevant_filenames = torch.load(core_bank_filenames_file)
    coreset_relevant_flat_features_bank = torch.load(coreset_relevant_flat_features_bank_file).to(device)
    coreset_features = torch.load(template_features_bank_coreset_file).to(device)
    print(f"Coreset cargado. Dimensión: {coreset_features.shape}")
except Exception as e:
    print(f"ERROR al cargar archivos del coreset: {e}. Asegúrate de que la Etapa 1 se ejecutó.")
    exit()

# Mover coreset a CPU para sklearn's NearestNeighbors
coreset_features_cpu = coreset_features.cpu().numpy()
# se calcula la distancia coseno == 1 - similitud coseno [0,1] 0 identico, 1 completamente diferente
nn_finder = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(coreset_features_cpu)
print("NearestNeighbors finder inicializado.")

# --- Cargar Modelo DINOv2 ---
print("Cargando modelo DINOv2...")
#featup_local_path = "/home/imercatoma/FeatUp"
#upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm, source='local').to(device)
#upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
featup_local_path = "/home/imercatoma/FeatUp"
upsampler = torch.hub.load(featup_local_path, 'dinov2', use_norm=use_norm, source='local').to(device)

dinov2_model = upsampler.model
dinov2_model.eval()
print("Modelo DINOv2 cargado.")

# --- Transformación de Imagen ---
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(),
    norm
])

# --- Carga del Modelo SAM2 (Ámbito Global) ---
checkpoint = "/home/imercatoma/sam2_repo_independent/checkpoints/sam2.1_hiera_small.pt"
model_cfg_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_model = None

try:
    print(f"Cargando modelo SAM2: {checkpoint} con config: {model_cfg_name}")
    loaded_sam2_model = build_sam2(model_cfg_name, checkpoint, device=device, apply_postprocessing=True)
    loaded_sam2_model.eval()
    sam2_model = loaded_sam2_model
    print("Modelo SAM2 cargado.")
except Exception as e:
    print(f"ERROR al cargar el modelo SAM2: {e}")
    sam2_model = None

# --- Función Principal para Puntuaciones de Anomalía ---
def get_anomaly_scores_for_image(image_path, model, image_transform, nn_finder_instance, H_prime, W_prime, device):
    try:
        query_img_pil = Image.open(image_path).convert("RGB")
        input_tensor = image_transform(query_img_pil).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error cargando/transformando imagen {os.path.basename(image_path)}: {e}")
        return None, None, None, None, None

    with torch.no_grad():
        features_lr = model(input_tensor)
        
    print("Shape de features_lr:", features_lr.shape)
    query_patches_flat = features_lr.squeeze(0).permute(1, 2, 0).reshape(-1, features_lr.shape[1]) # shape (H_prime * W_prime, C)
    print("Primeros 5 patches_flat:")
    print(query_patches_flat[:5])
    query_patches_flat_cpu = query_patches_flat.cpu().numpy()
    
    distances_to_nn, _ = nn_finder_instance.kneighbors(query_patches_flat_cpu)
    
    print("Shape de distances_to_nn:", distances_to_nn.shape)
    print("Distancias:", distances_to_nn[5:].flatten())
    print("Índices:", np.argsort(distances_to_nn, axis=0)[:5].flatten())
    print("Máximo de distances_to_nn:", np.max(distances_to_nn))
    print("Mínimo de distances_to_nn:", np.min(distances_to_nn))
    
    
    patch_anomaly_scores = distances_to_nn.flatten()
    print("Primeros 5 patch_anomaly_scores:", patch_anomaly_scores[:5])
    sorted_patch_anomaly_scores = np.sort(patch_anomaly_scores)[::-1]
    print("Primeros 5 sorted_patch_anomaly_scores:", sorted_patch_anomaly_scores[:20])
    print("Máximo de sorted_patch_anomaly_scores:", np.max(sorted_patch_anomaly_scores))
    print("Mínimo de sorted_patch_anomaly_scores:", np.min(sorted_patch_anomaly_scores))
    
    anomaly_map_lr = patch_anomaly_scores.reshape(H_prime, W_prime)
    anomaly_map_lr_tensor = torch.from_numpy(anomaly_map_lr).unsqueeze(0).unsqueeze(0).to(device)
    anomaly_map_upsampled = F.interpolate(anomaly_map_lr_tensor, size=(input_size, input_size), mode='bilinear', align_corners=False)
    print("Shape de anomaly_map_upsampled:", anomaly_map_upsampled.shape)
    anomaly_map_upsampled = anomaly_map_upsampled.squeeze().cpu().numpy()
    print("Primeros 5 valores de anomaly_map_upsampled (aplanado):")
    print(anomaly_map_upsampled.flatten()[:5])
    print("Máximo de anomaly_map_upsampled:", np.max(anomaly_map_upsampled))
    print("Mínimo de anomaly_map_upsampled:", np.min(anomaly_map_upsampled))
    
    anomaly_map_smoothed = gaussian_filter(anomaly_map_upsampled, sigma=4.0)

    if anomaly_map_smoothed.max() == anomaly_map_smoothed.min():
        anomaly_map_final = np.zeros_like(anomaly_map_smoothed, dtype=float)
    else:
        anomaly_map_final = (anomaly_map_smoothed - anomaly_map_smoothed.min()) / (anomaly_map_smoothed.max() - anomaly_map_smoothed.min() + 1e-8)

    return patch_anomaly_scores, sorted_patch_anomaly_scores, query_img_pil, anomaly_map_final, features_lr

# --- Funciones de Métricas ---
def calculate_rms(data):
    return np.sqrt(np.mean(data**2))

def calculate_mad(data):
    return median_abs_deviation(data)

def calculate_median(data):
    return np.median(data)

def calculate_quartile(data, q=25):
    return np.percentile(data, q)

# --- Funciones de Filtrado de Anomalías ---
def calculate_spatial_variance_of_top_patches(patch_anomaly_scores, top_percentage=5.5):
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0:
        return np.nan

    num_patches = patch_anomaly_scores.size
    num_top = max(1, int(num_patches * top_percentage / 100))
    top_patch_indices = np.argsort(patch_anomaly_scores)[-num_top:]
    row_coords = top_patch_indices // W_prime
    col_coords = top_patch_indices % W_prime

    std_rows = np.std(row_coords) if len(row_coords) > 1 else 0.0
    std_cols = np.std(col_coords) if len(col_coords) > 1 else 0.0
    return std_rows + std_cols

def calculate_active_patches_count_relative_threshold(patch_anomaly_scores, relative_threshold_percentage):
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0: return 0
    max_val_in_image = np.max(patch_anomaly_scores)
    if max_val_in_image == 0: return 0
    threshold_val = max_val_in_image * relative_threshold_percentage
    return len(patch_anomaly_scores[patch_anomaly_scores > threshold_val])

def calculate_top_percent_average_anomaly(patch_anomaly_scores, top_percent=1):
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0: return 0.0
    num_patches = patch_anomaly_scores.size
    num_top = max(1, int(num_patches * top_percent / 100))
    sorted_scores = np.sort(patch_anomaly_scores)[::-1]
    return np.mean(sorted_scores[:num_top])

# --- Generar y Guardar Mapas de Calor ---
def generate_and_save_heatmap(image_original_pil, anomaly_map_final, sorted_patch_anomaly_scores, save_path, image_name_for_title):
    num_patches = len(sorted_patch_anomaly_scores)
    num_top_for_q_score = max(1, int(num_patches * 0.01))
    q_score = np.mean(sorted_patch_anomaly_scores[:num_top_for_q_score])
    anomalia_estructural = q_score > 0.27

    print(f"Q-score: {q_score:.4f}. Anomalía estructural: {'Sí' if anomalia_estructural else 'No'}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_original_pil)
    plt.title(f'Imagen Original: {image_name_for_title}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(anomaly_map_final, cmap='jet')
    plt.title(f'Mapa de Anomalía (Q-score: {q_score:.2f})')
    plt.colorbar(label='Puntuación de Anomalía Normalizada')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Mapa de calor de anomalías guardado en: {save_path}")
    plt.close()
    return anomaly_map_final, q_score, anomalia_estructural

base_image_name = os.path.basename(query_image_path)
print(f"\n--- Procesando imagen: {base_image_name} ---")

RELATIVE_ACTIVE_PATCH_THRESHOLD_PERCENTAGE = 0.80

current_patch_anomaly_scores, current_sorted_patch_anomaly_scores, query_img_pil, anomaly_map_final_for_regions, query_lr_features = get_anomaly_scores_for_image(
    query_image_path, dinov2_model, transform, nn_finder, H_prime, W_prime, device
)


print("Variables obtenidas:")
# Imprimir los 5 primeros valores de los términos solicitados
if current_patch_anomaly_scores is not None:
    print("Primeros 5 valores de current_patch_anomaly_scores:")
    print(current_patch_anomaly_scores[:5])

if current_sorted_patch_anomaly_scores is not None:
    print("Primeros 5 valores de current_sorted_patch_anomaly_scores:")
    print(current_sorted_patch_anomaly_scores[:5])

if query_img_pil is not None:
    print("Shape de query_img_pil:")
    print(query_img_pil.size)  # PIL images use (width, height)

if anomaly_map_final_for_regions is not None:
    print("Primeros 5 valores de anomaly_map_final_for_regions (aplanado):")
    print(anomaly_map_final_for_regions.flatten()[:5])

if query_lr_features is not None:
    print("Primeras 5 características de query_lr_features:")
    print(query_lr_features.reshape(-1, query_lr_features.shape[-1])[:5])

# Mostrar las 5 primeras características de query_lr_features
print("Primeras 5 características de query_lr_features:")
print(query_lr_features.reshape(-1, query_lr_features.shape[-1])[:5])
print("Máximo de query_lr_features:", torch.max(query_lr_features).item())
print("Mínimo de query_lr_features:", torch.min(query_lr_features).item())






# Generar y guardar heatmap
heatmap_filename = f"heat_{base_image_name}"
individual_heatmap_save_path = os.path.join(HEATMAPS_SAVE_DIR, heatmap_filename)
current_anomaly_map_final, current_q_score, current_anomalia_estructural = \
    generate_and_save_heatmap(query_img_pil, anomaly_map_final_for_regions,
                                current_sorted_patch_anomaly_scores,
                                individual_heatmap_save_path, base_image_name.replace(".png", ""))

# Calcular métricas para la imagen actual
min_val = np.min(current_sorted_patch_anomaly_scores)
max_val = np.max(current_sorted_patch_anomaly_scores)
normalized_data = (current_sorted_patch_anomaly_scores - min_val) / (max_val - min_val + 1e-8) if max_val != min_val else np.zeros_like(current_sorted_patch_anomaly_scores)

A_rms = calculate_rms(normalized_data)
B_mad = calculate_mad(normalized_data)
C_median = calculate_median(normalized_data)
D_q1_normalized = calculate_quartile(normalized_data, q=25)

dist_rms_mad = A_rms - B_mad
dist_rms_median = A_rms - C_median
dist_rms_q1 = A_rms - D_q1_normalized
spatial_var = calculate_spatial_variance_of_top_patches(current_patch_anomaly_scores)
active_count = calculate_active_patches_count_relative_threshold(current_patch_anomaly_scores, relative_threshold_percentage=RELATIVE_ACTIVE_PATCH_THRESHOLD_PERCENTAGE)
top_1_avg = calculate_top_percent_average_anomaly(current_patch_anomaly_scores, top_percent=1)

# Lógica de clasificación
classification = 0
if top_1_avg >= 0.30:
    classification = 1
    print(f"Clasificación: ANOMALÍA GRANDE (Top 1% Avg: {top_1_avg:.4f} >= 0.30)")
elif 0.17 <= top_1_avg < 0.30:
    print(f"Clasificación: Evaluación de anomalía leve/buena (Top 1% Avg: {top_1_avg:.4f})")
    initial_classification_based_on_active_patches = 0
    if active_count > 5:
        initial_classification_based_on_active_patches = 1
        print(f"-> ANOMALÍA LEVE (Parches Activos: {active_count} > 5)")
        classification = 1
    else:
        print(f"-> Parches Activos ({active_count}) <= 5. Evaluando 'buena'.")
        initial_classification_based_on_active_patches = 0
        if dist_rms_median <= 0.055:
            print(f"-> Condición Buena II (RMS - Mediana <= 0.055): True ({dist_rms_median:.4f})")
            cond_I_met = spatial_var >= 5.5
            print(f"-> Condición Buena I (Varianza Espacial >= 5.5): {'True' if cond_I_met else 'False'} ({spatial_var:.2f})")
            cond_III_met = dist_rms_mad >= 0.21
            print(f"-> Condición Buena III (RMS - MAD >= 0.21): {'True' if cond_III_met else 'False'} ({dist_rms_mad:.4f})")
            if cond_I_met or cond_III_met:
                classification = 0
                print(f"-> IMAGEN BUENA")
            else:
                classification = initial_classification_based_on_active_patches
                print(f"-> {'ANOMALÍA LEVE' if classification == 1 else 'IMAGEN BUENA'} (Revertiendo a Parches Activos)")
        else:
            classification = 1
            print(f"-> ANOMALÍA LEVE")

print(f"Clasificación Final para {base_image_name}: {'Anómala' if classification == 1 else 'Buena'}")

# --- Detección y visualización de regiones de anomalía "fuertes" ---
if classification == 1:
    start_time_region_detection = time.time()
    print("\n  ** Clasificada como ANÓMALA. Buscando regiones fuertes... **")
    strong_anomaly_region_threshold = 0.75
    binary_strong_anomaly_map = anomaly_map_final_for_regions > strong_anomaly_region_threshold
    
    if not np.any(binary_strong_anomaly_map):
        print(f"    No se encontraron píxeles por encima del umbral de {strong_anomaly_region_threshold}.")
    else:
        labeled_anomaly_regions = measure.label(binary_strong_anomaly_map)
        region_properties = measure.regionprops(labeled_anomaly_regions)
        detected_strong_anomaly_regions = []
        min_region_pixel_area = 50
        original_img_width, original_img_height = query_img_pil.size
        scale_x = original_img_width / input_size
        scale_y = original_img_height / input_size

        for region in region_properties:
            if region.area >= min_region_pixel_area:
                min_y, min_x, max_y, max_x = region.bbox
                scaled_min_x = int(np.clip(min_x * scale_x, 0, original_img_width))
                scaled_min_y = int(np.clip(min_y * scale_y, 0, original_img_height))
                scaled_max_x = int(np.clip(max_x * scale_x, 0, original_img_width))
                scaled_max_y = int(np.clip(max_y * scale_y, 0, original_img_height))
                region_width = scaled_max_x - scaled_min_x
                region_height = scaled_max_y - scaled_min_y
                if region_width > 0 and region_height > 0:
                    detected_strong_anomaly_regions.append({
                        'bbox': (scaled_min_x, scaled_min_y, region_width, region_height),
                        'area_pixels': region.area
                    })

        if detected_strong_anomaly_regions:
            plt.figure(figsize=(10, 8))
            plt.imshow(query_img_pil)
            plt.title(f'Imagen Anómala con Regiones Fuertes: {base_image_name.replace(".png", "")}')
            plt.axis('off')
            ax = plt.gca()
            for region_info in detected_strong_anomaly_regions:
                bbox = region_info['bbox']
                if bbox[2] > 0 and bbox[3] > 0:
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                             linewidth=3, edgecolor='lime', facecolor='none', linestyle='-', alpha=0.9)
                    ax.add_patch(rect)
            ax.add_patch(patches.Rectangle((0,0), 0.1, 0.1, linewidth=3, edgecolor='lime', facecolor='none', linestyle='-', alpha=0.9, label=f'Regiones Anómalas Fuertes'))
            plt.legend()
            strong_regions_overlay_output_filename = os.path.join(ANOMALY_REGIONS_SAVE_DIR, f'anomaly_regions_{base_image_name}')
            plt.tight_layout()
            plt.savefig(strong_regions_overlay_output_filename)
            plt.close()
            print(f"    Plot de regiones anómalas fuertes guardado en: {strong_regions_overlay_output_filename}")
        else:
            print("    No se detectaron regiones válidas para dibujar.")
    end_time_region_detection = time.time()
    print(f"  Tiempo para detección de regiones: {end_time_region_detection - start_time_region_detection:.4f} segundos.")
else:
    print(f"  Clasificada como BUENA. No se dibujarán regiones anómalas.")

directorio_imagenes = TRAIN_GOOD_DIR
plot_save_directory_on_server = PLOT_SAVE_ROOT_DIR

# --- Función para buscar imágenes similares usando KNN ---
def buscar_imagenes_similares_knn(query_feature_map, pre_flattened_features_bank, k=3, nombres_archivos=None):
    query_feat_flatten = query_feature_map.flatten().cpu().numpy()
    features_bank_for_knn = pre_flattened_features_bank.cpu().numpy() if isinstance(pre_flattened_features_bank, torch.Tensor) else pre_flattened_features_bank

    start_time_knn_dist = time.time()
    distances = euclidean_distances([query_feat_flatten], features_bank_for_knn)
    nearest_indices = np.argsort(distances[0])[:k]
    end_time_knn_dist = time.time()
    print(f"Tiempo para calcular distancias KNN: {end_time_knn_dist - start_time_knn_dist:.4f} segundos")

    imagenes_similares = []
    rutas_imagenes_similares = []
    if nombres_archivos:
        for idx in nearest_indices:
            imagenes_similares.append(nombres_archivos[idx])
            rutas_imagenes_similares.append(os.path.join(directorio_imagenes, nombres_archivos[idx]))
    else: # Fallback if no filenames provided (less common for this use case)
        for idx in nearest_indices:
            imagenes_similares.append(f"Imagen_Banco_{idx:03d}.png")
            rutas_imagenes_similares.append(os.path.join(directorio_imagenes, f"Imagen_Banco_{idx:03d}.png"))
    return imagenes_similares, rutas_imagenes_similares, end_time_knn_dist

# --- Búsqueda KNN ---
print("\nBuscando imágenes similares usando el banco pre-aplanado del Coreset...")
imagenes_similares, rutas_imagenes_similares, time_knn_dist = buscar_imagenes_similares_knn(
    query_lr_features, coreset_relevant_flat_features_bank, nombres_archivos=coreset_relevant_filenames
)
print("Imágenes similares:", imagenes_similares)

# --- Visualización de imágenes similares ---
plt.figure(figsize=(15, 5))
plt.subplot(1, len(rutas_imagenes_similares) + 1, 1)
plt.imshow(query_img_pil)
plt.title(f'Consulta:\n{base_image_name}')
plt.axis('off')

for j, ruta_imagen_similar in enumerate(rutas_imagenes_similares):
    try:
        img_similar = Image.open(ruta_imagen_similar).convert('RGB')
        plt.subplot(1, len(rutas_imagenes_similares) + 1, j + 2)
        plt.imshow(img_similar)
        plt.title(f'Vecino {j + 1}\n({os.path.basename(ruta_imagen_similar)})')
        plt.axis('off')
    except Exception as e:
        print(f"Error al cargar imagen similar {os.path.basename(ruta_imagen_similar)}: {e}")
        plt.subplot(1, len(rutas_imagenes_similares) + 1, j + 2)
        plt.text(0.5, 0.5, "Error de Carga", ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'Vecino {j + 1}\n(Error)')
        plt.axis('off')

output_similar_plot_filename = os.path.join(plot_save_directory_on_server, f'similar_images_plot_{base_image_name}')
plt.tight_layout()
plt.savefig(output_similar_plot_filename)
plt.close()
print(f"Plot de imágenes similares guardado en: {output_similar_plot_filename}")

# --- Aplicar FeatUp para obtener características de alta resolución ---
def apply_featup_hr(image_path, featup_upsampler, image_transform, device):
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        lr_feats = featup_upsampler.model(image_tensor)
        hr_feats = featup_upsampler(image_tensor)
    return lr_feats.cpu(), hr_feats.cpu()

# Características de la imagen de consulta
input_query_tensor_original = transform(Image.open(query_image_path).convert("RGB")).unsqueeze(0).to(device)
query_lr_feats_featup, query_hr_feats = apply_featup_hr(query_image_path, upsampler, transform, device)

plot_feats(unnorm(input_query_tensor_original)[0], query_lr_feats_featup[0], query_hr_feats[0])
fig_query_feats = plt.gcf()
fig_query_feats.suptitle(f'Características FeatUp: {base_image_name.replace(".png", "")}')
output_query_feat_plot_filename = os.path.join(FEATUP_PLOTS_DIR, f'featup_query_image_features_plot_{base_image_name}')
plt.tight_layout()
fig_query_feats.savefig(output_query_feat_plot_filename)
plt.close(fig_query_feats)
print(f"Plot de características FeatUp (consulta) guardado en: {output_query_feat_plot_filename}")

# Características de las imágenes similares
similar_hr_feats_list = []
for j, similar_image_path in enumerate(rutas_imagenes_similares):
    input_similar_tensor_original = transform(Image.open(similar_image_path).convert("RGB")).unsqueeze(0).to(device)
    similar_lr_feats, similar_hr_feats = apply_featup_hr(similar_image_path, upsampler, transform, device)
    similar_hr_feats_list.append(similar_hr_feats)

    plot_feats(unnorm(input_similar_tensor_original)[0], similar_lr_feats[0], similar_hr_feats[0])
    fig_similar_feats = plt.gcf()
    fig_similar_feats.suptitle(f'Características FeatUp Vecino {j + 1}: {os.path.basename(similar_image_path).replace(".png", "")}')
    output_similar_feat_plot_filename = os.path.join(FEATUP_PLOTS_DIR, f'featup_similar_image_{j + 1}_features_plot_{base_image_name}')
    plt.tight_layout()
    fig_similar_feats.savefig(output_similar_feat_plot_filename)
    plt.close(fig_similar_feats)
    print(f"Plot de características FeatUp (vecino {j + 1}) guardado en: {output_similar_feat_plot_filename}")

### Aplicando Máscaras SAM query y similares

print(f"\nIniciando SAM para la imagen anómala: {base_image_name}")
start_time_sam = time.time()

# --- Funciones Auxiliares de Visualización ---
def show_mask(mask, ax, random_color=False, borders=True):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image_alpha = np.zeros((h, w, 4), dtype=np.float32)
    mask_image_alpha[mask > 0] = color
    if borders:
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_image = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=2)
        contour_mask = (contour_image.astype(np.float32) / 255.0).sum(axis=-1) > 0
        mask_image_alpha[contour_mask > 0, :3] = 1.0
        mask_image_alpha[contour_mask > 0, 3] = 0.5
    ax.imshow(mask_image_alpha)

def show_points(coords, labels, ax, marker_size=375):
    ax.scatter(coords[labels==1][:, 0], coords[labels==1][:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(coords[labels==0][:, 0], coords[labels[0]==0][:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_masks_grid(image, masks, points=None, plot_title="Generated Masks", ax=None, num_masks=0):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    if points is not None:
        show_points(points, np.ones(points.shape[0], dtype=int), ax, marker_size=50)
    for mask_data in masks:
        show_mask(mask_data["segmentation"], ax, random_color=True)
    ax.set_title(f"{plot_title} (Masks: {num_masks})", fontsize=18)
    ax.axis('off')
# --- Fin Funciones Auxiliares ---

try:
    image_for_sam_np = np.array(Image.open(query_image_path).convert("RGB"))
    print(f"Dimensiones imagen SAM: {image_for_sam_np.shape}")
except Exception as e:
    print(f"Error procesando imagen para SAM: {e}. Saltando SAM.")
    sam2_model = None

if sam2_model is not None:
    points_grid_density = 16
    min_mask_area_pixels = 200.0
    max_mask_area_pixels = 450000.0

    mask_generator_query = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_grid_density,
        points_per_batch=256,
        pred_iou_thresh=0.48,
        stability_score_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=min_mask_area_pixels,
    )

    print(f"Generando máscaras para consulta con grid de {points_grid_density}x{points_grid_density} puntos...")
    masks_data_query_image = mask_generator_query.generate(image_for_sam_np)
    masks_data_query_image = [m for m in masks_data_query_image if m['area'] <= max_mask_area_pixels]
    print(f"Máscaras generadas para consulta: {len(masks_data_query_image)}.")

    mask_generator_similar = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_grid_density,
        points_per_batch=256,
        pred_iou_thresh=0.48,
        stability_score_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=min_mask_area_pixels,
    )

    print("\nGenerando máscaras SAM para imágenes similares...")
    similar_masks_raw_list = []
    for j, similar_image_path in enumerate(rutas_imagenes_similares):
        try:
            image_np_similar_for_sam = np.array(Image.open(similar_image_path).convert('RGB'))
            print(f"--- Procesando vecino {j+1}: {os.path.basename(similar_image_path)} ---")
            current_similar_masks_data = mask_generator_similar.generate(image_np_similar_for_sam)
            current_similar_masks_data = [m for m in current_similar_masks_data if m['area'] <= max_mask_area_pixels]
            similar_masks_raw_list.append(current_similar_masks_data)
            print(f"Máscaras generadas para vecino {j+1}: {len(current_similar_masks_data)}.")
        except Exception as e:
            print(f"Error procesando imagen similar {os.path.basename(similar_image_path)} para SAM: {e}")

    end_time_sam = time.time()
    print(f"Tiempo total de ejecución de SAM: {end_time_sam - start_time_sam:.4f} segundos.")

    print("\nGenerando visualización combinada de imágenes segmentadas...")
    combined_plots_directory = os.path.join(PLOT_SAVE_ROOT_DIR, "combined_segmented_plots")
    os.makedirs(combined_plots_directory, exist_ok=True)

    def plot_combined_segmented(query_original_path, query_masks, similar_original_paths, similar_masks_list, output_dir, current_image_name):
        num_similar = len(similar_original_paths)
        if num_similar == 0: return

        total_subplots = 2 + num_similar
        fig, axes = plt.subplots(1, total_subplots, figsize=(5 * total_subplots, 6))

        try:
            query_img_orig = Image.open(query_original_path).convert('RGB')
            axes[0].imshow(query_img_orig)
            axes[0].set_title(f'Consulta Original:\n{current_image_name.replace(".png", "")}')
            axes[0].axis('off')
            show_masks_grid(np.array(query_img_orig), query_masks, plot_title=f'Consulta Segmentada', ax=axes[1], num_masks=len(query_masks))
        except Exception as e:
            print(f"Error al graficar imagen de consulta original/segmentada: {e}")
            axes[0].text(0.5, 0.5, "Error", ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Consulta Original (Error)'); axes[0].axis('off')
            axes[1].text(0.5, 0.5, "Error", ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Consulta Segmentada (Error)'); axes[1].axis('off')

        for j, similar_path in enumerate(similar_original_paths):
            if j + 2 >= total_subplots: break
            try:
                similar_img_orig = Image.open(similar_path).convert('RGB')
                current_similar_masks = similar_masks_list[j] if j < len(similar_masks_list) else []
                show_masks_grid(np.array(similar_img_orig), current_similar_masks,
                                plot_title=f'Vecino {j+1} Segmentado\n({os.path.basename(similar_path)})',
                                ax=axes[j + 2], num_masks=len(current_similar_masks))
            except Exception as e:
                print(f"Error al graficar imagen similar {os.path.basename(similar_path)}: {e}")
                axes[j + 2].text(0.5, 0.5, "Error", ha='center', va='center', transform=axes[j + 2].transAxes)
                axes[j + 2].set_title(f'Vecino {j+1} (Error)'); axes[j + 2].axis('off')

        plt.tight_layout()
        output_filename = os.path.join(output_dir, f'combined_query_and_similar_segmented_{current_image_name}')
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"Plot combinado de imágenes segmentadas guardado en: {output_filename}")

    plot_combined_segmented(
        query_image_path,
        masks_data_query_image,
        rutas_imagenes_similares,
        similar_masks_raw_list,
        combined_plots_directory,
        base_image_name
    )
    
print(f"La imagen {base_image_name} fue clasificada como BUENA o el modelo SAM no se pudo cargar. No se generarán máscaras SAM.")
print("\nAnálisis de detección de anomalías para una sola imagen completado.")

# --- Implementación del punto 3.4.3. Object Feature Map ---
import torch.nn.functional as F # Importa F para F.interpolate
from sklearn.decomposition import PCA

def process_masks_to_object_feature_maps(raw_masks, hr_feature_map, target_h, target_w, sam_processed_image_shape):
    """
    Procesa una lista de máscaras de SAM para obtener mapas de características de objeto.
    Args:
        raw_masks (list): Lista de diccionarios de máscaras crudas de SAM.
                          Cada dict tiene una clave 'segmentation' (np.ndarray booleana).
        hr_feature_map (torch.Tensor): Mapa de características de alta resolución (C, 8H', 8W').
                                        Debe ser de la imagen correspondiente (query o reference).
                                        Asegúrate de que ya esté en el dispositivo correcto.
        target_h (int): Altura objetivo para la máscara escalada (8H').
        target_w (int): Ancho objetivo para la máscara escalada (8W').
        sam_processed_image_shape (tuple): La forma (H, W, C) de la imagen a la que SAM se aplicó
                                            para generar las máscaras (ej. (1024, 1024, 3)).
                                            Esto es crucial para escalar correctamente la máscara.
    Returns:
        torch.Tensor: Tensor de mapas de características de objeto (M, C, 8H', 8W').
                      Si no hay máscaras, devuelve un tensor vacío (0, C, 8H', 8W').
    """
    if not raw_masks:
        print("Advertencia: No se encontraron máscaras para procesar. Devolviendo tensor vacío.")
        C_dim = hr_feature_map.shape[0] if hr_feature_map.ndim >=3 else 0
        return torch.empty(0, C_dim, target_h, target_w, device=hr_feature_map.device)

    object_feature_maps_list = []
    C_dim = hr_feature_map.shape[0] # Número de canales de las características HR

    for mask_info in raw_masks:
        # Convertir la máscara booleana de numpy a tensor float y añadir dimensiones de lote y canal
        mask_np = mask_info['segmentation'].astype(np.float32)
        mask_tensor_original_res = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0) # (1, 1, H_orig, W_orig)
        # Mover la máscara al mismo dispositivo que el mapa de características HR
        mask_tensor_original_res = mask_tensor_original_res.to(hr_feature_map.device)

        # 1. Escalar la máscara a (8H', 8W') usando interpolación bilineal
        scaled_mask = F.interpolate(mask_tensor_original_res,
                                     size=(target_h, target_w),
                                     mode='bilinear',
                                     align_corners=False)
        # Opcional: Binarizar la máscara después del escalado si se requiere una máscara estricta (0 o 1)
        scaled_mask = (scaled_mask > 0.5).float()

        # 2. Multiplicación elemento a elemento con el mapa de características HR
        if hr_feature_map.ndim == 3:
            hr_feature_map_with_batch = hr_feature_map.unsqueeze(0) # -> (1, C, H, W)
        else: # Si ya es (1, C, H, W)
            hr_feature_map_with_batch = hr_feature_map

        object_feature_map_i = scaled_mask * hr_feature_map_with_batch
        object_feature_maps_list.append(object_feature_map_i)

    # Concatenar todos los mapas de características de objeto
    final_object_feature_maps = torch.cat(object_feature_maps_list, dim=0) # (M, C, 8H', 8W')

    return final_object_feature_maps

# --- Visualización de Mapas de Características de Objeto ---
def visualize_object_feature_map(original_image_path, sam_mask_info, hr_feature_map_tensor,
                                   object_feature_map_tensor, target_h, target_w,
                                   plot_save_dir, plot_filename_prefix, mask_idx,
                                   sam_processed_image_shape):
    """
    Genera y guarda una visualización de un mapa de características de objeto.
    Muestra la imagen original, la máscara de SAM y el mapa de características de objeto.
    """
    try:
        original_img = Image.open(original_image_path).convert("RGB")
        sam_mask_np = sam_mask_info['segmentation']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Imagen Original
        axes[0].imshow(original_img)
        axes[0].set_title(f'Imagen Original\n{os.path.basename(original_image_path)}')
        axes[0].axis('off')

        # Plot 2: Máscara SAM (escalada para visualización si es necesario, pero manteniendo la forma original)
        # We need to scale the SAM mask to the input_size for direct overlay if original_img is resized,
        # but the mask itself comes from the SAM processed image which might be 1024x1024.
        # For display simplicity, we'll just show the original mask over the original image,
        # ensuring the aspect ratio aligns.
        mask_display = sam_mask_np # Boolean mask
        axes[1].imshow(original_img) # Overlay on original
        # For plotting mask, we scale it to match the original image's aspect ratio/size if necessary for correct overlay
        # Since SAM masks are usually for specific input sizes (e.g., 1024x1024), we should ensure it fits.
        # However, for simplicity here, we assume the mask is compatible or will be interpolated by imshow.
        show_mask(mask_display, axes[1], random_color=False, borders=True) # Use the show_mask helper
        axes[1].set_title(f'Máscara SAM {mask_idx}')
        axes[1].axis('off')

        # Plot 3: Object Feature Map (visualización de PCA)
        # Reshape C, H, W to (H*W, C) for PCA
        if object_feature_map_tensor.numel() == 0:
            axes[2].text(0.5, 0.5, "No hay características de objeto", ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Mapa de Características de Objeto (Vacío)')
            axes[2].axis('off')
        else:
            ofm_cpu = object_feature_map_tensor.squeeze().cpu().numpy() # Remove batch dim if present
            if ofm_cpu.ndim == 3: # C, H, W
                C, H, W = ofm_cpu.shape
                ofm_reshaped = ofm_cpu.transpose(1, 2, 0).reshape(-1, C) # H*W, C

                if C > 3: # Apply PCA if more than 3 channels
                    pca = PCA(n_components=3)
                    ofm_pca = pca.fit_transform(ofm_reshaped)
                    # Normalize PCA results to [0, 1] for image display
                    ofm_pca_normalized = (ofm_pca - ofm_pca.min()) / (ofm_pca.max() - ofm_pca.min() + 1e-8)
                    ofm_display = ofm_pca_normalized.reshape(H, W, 3)
                    axes[2].imshow(ofm_display)
                    axes[2].set_title(f'Mapa de Características de Objeto (PCA)\nMáscara {mask_idx}')
                else: # If 1, 2, or 3 channels, display directly (grayscale or RGB)
                    if C == 1:
                        ofm_display = ofm_cpu.squeeze()
                        axes[2].imshow(ofm_display, cmap='viridis')
                    elif C == 3:
                        ofm_display = ofm_cpu.transpose(1, 2, 0) # H, W, C
                        ofm_display_norm = (ofm_display - ofm_display.min()) / (ofm_display.max() - ofm_display.min() + 1e-8)
                        axes[2].imshow(ofm_display_norm)
                    else: # 2 channels, or other, might not display well as RGB. Use grayscale of first channel.
                        ofm_display = ofm_cpu[0]
                        axes[2].imshow(ofm_display, cmap='viridis')
                    axes[2].set_title(f'Mapa de Características de Objeto\nMáscara {mask_idx}')
            else: # If the object_feature_map_tensor somehow resulted in a non-3D tensor for a single mask
                axes[2].text(0.5, 0.5, "Formato de características de objeto inesperado", ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_title('Mapa de Características de Objeto (Error)')

            axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(plot_save_dir, f"{plot_filename_prefix}_mask_{mask_idx}.png")
        plt.savefig(save_path)
        plt.close(fig)
        # print(f"Visualización del mapa de características de objeto guardada en: {save_path}")

    except Exception as e:
        print(f"Error al visualizar el mapa de características de objeto para máscara {mask_idx} de {os.path.basename(original_image_path)}: {e}")

# --- Aplicar el proceso a la imagen de consulta y a las imágenes de referencia ---

print("\n--- Generando Mapas de Características de Objeto ---")

# Dimensiones objetivo para las máscaras después de escalar (8H', 8W')
TARGET_MASK_H = 8 * H_prime # 8 * 16 = 128
TARGET_MASK_W = 8 * W_prime # 8 * 16 = 128

# Para la imagen de consulta (Iq)
fobj_q = process_masks_to_object_feature_maps(
    masks_data_query_image,
    query_hr_feats.squeeze(0), # Pasamos (C, 8H', 8W') para que la función maneje el batch
    TARGET_MASK_H,
    TARGET_MASK_W,
    image_for_sam_np.shape # Pasamos la forma real de la imagen que SAM procesó
).to(device) # Mover a la GPU si no está ya

print(f"Dimensiones de fobj_q (Mapas de Características de Objeto de Iq): {fobj_q.shape}") # Esperado (M, 384, 128, 128)

# Para las imágenes de referencia (Ir)
all_fobj_r_list = [] # Para almacenar fobj_r para cada imagen similar
for i, similar_hr_feats in enumerate(similar_hr_feats_list):
    current_similar_masks_raw = similar_masks_raw_list[i]
    # Necesitamos obtener la forma original de la imagen similar para SAM
    img_similar_pil = Image.open(rutas_imagenes_similares[i]).convert('RGB') # Cargar de nuevo para obtener su forma
    image_np_similar_for_sam_shape = np.array(img_similar_pil).shape

    fobj_r_current = process_masks_to_object_feature_maps(
        current_similar_masks_raw,
        similar_hr_feats.squeeze(0), # Pasamos (C, 8H', 8W')
        TARGET_MASK_H,
        TARGET_MASK_W,
        image_np_similar_for_sam_shape # Pasamos la forma real de la imagen que SAM procesó
    ).to(device) # Mover a la GPU
    all_fobj_r_list.append(fobj_r_current)
    print(f"Dimensiones de fobj_r para vecino {i+1}: {fobj_r_current.shape}") # Esperado (N, 384, 128, 128)
    # Imprimir el tipo de cada elemento en all_fobj_r_list
    print("\nTipos de los elementos en all_fobj_r_list:")
    for idx, fobj_r in enumerate(all_fobj_r_list):
        print(f"Vecino {idx + 1}: Tipo de fobj_r:", type(fobj_r))
print("\nProceso de 'Object Feature Map' completado. ¡Ahora tienes los fobj_q y fobj_r listos!")


# --- Directorio para guardar plots de Mapas de Características de Objeto ---
OFM_PLOTS_DIR = os.path.join(PLOT_SAVE_ROOT_DIR, "object_feature_map_plots")
os.makedirs(OFM_PLOTS_DIR, exist_ok=True)
print(f"\nLos plots de Mapas de Características de Objeto se guardarán en: {OFM_PLOTS_DIR}")

# # Visualización para la imagen de consulta (Iq)
# print("\nGenerando visualizaciones de Mapas de Características de Objeto para la consulta...")
# for i, mask_info in enumerate(masks_data_query_image):
#     # fobj_q es (M, C, H, W). Necesitamos una máscara a la vez.
#     if i < fobj_q.shape[0]: # Asegurarse de que tenemos un OFM para esta máscara
#         visualize_object_feature_map(
#             query_image_path,
#             mask_info,
#             query_hr_feats, # Pasamos el HR feature map completo
#             fobj_q[i].unsqueeze(0), # Pasamos solo el OFM de la máscara actual, con batch dim para la función
#             TARGET_MASK_H,
#             TARGET_MASK_W,
#             OFM_PLOTS_DIR,
#             f"query_{base_image_name.replace('.png', '')}",
#             i,
#             image_for_sam_np.shape
#         )
#     else:
#         print(f"Advertencia: No se encontró OFM para la máscara de consulta {i}.")

# # Visualización para las imágenes de referencia (Ir)
# print("\nGenerando visualizaciones de Mapas de Características de Objeto para los vecinos...")
# for i, similar_image_path in enumerate(rutas_imagenes_similares):
#     current_similar_masks_raw = similar_masks_raw_list[i]
#     current_similar_hr_feats = similar_hr_feats_list[i]
#     current_fobj_r = all_fobj_r_list[i]
#     img_similar_pil_for_shape = Image.open(similar_image_path).convert('RGB')
#     image_np_similar_for_sam_shape = np.array(img_similar_pil_for_shape).shape

#     if not current_fobj_r.numel() == 0: # Solo procesar si hay OFMs generados para este vecino
#         for j, mask_info in enumerate(current_similar_masks_raw):
#             if j < current_fobj_r.shape[0]: # Asegurarse de que tenemos un OFM para esta máscara
#                 visualize_object_feature_map(
#                     similar_image_path,
#                     mask_info,
#                     current_similar_hr_feats,
#                     current_fobj_r[j].unsqueeze(0), # OFM de la máscara actual
#                     TARGET_MASK_H,
#                     TARGET_MASK_W,
#                     OFM_PLOTS_DIR,
#                     f"neighbor_{i+1}_{os.path.basename(similar_image_path).replace('.png', '')}",
#                     j,
#                     image_np_similar_for_sam_shape
#                 )
#             else:
#                 print(f"Advertencia: No se encontró OFM para la máscara {j} del vecino {i+1}.")
#     else:
#         print(f"No se generaron OFMs para el vecino {i+1} ({os.path.basename(similar_image_path)}), saltando visualización.")

# print("\nVisualización de Mapas de Características de Objeto completada.")

# -----------3.5.2 Object matching module-----------------
## Matching
# --- Definición de la función show_anomalies_on_image ---
def show_anomalies_on_image(image_np, masks, anomalous_info, alpha=0.5, save_path=None):

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)

    for obj_id, similarity in anomalous_info: # Iterate through (id, similarity) tuples
        # Extraer la máscara binaria real
        mask = masks[obj_id]['segmentation']
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # Crear máscara en rojo
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask > 0] = [255, 0, 0]
        plt.imshow(colored_mask, alpha=alpha)

        # Calcular centroide para colocar el texto
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
            
            # Create text with index and percentage
            text_label = f"{obj_id} ({similarity*100:.2f}%)"
            plt.text(cx, cy, text_label, color='white', fontsize=10, fontweight='bold', ha='center', va='center',
                     bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.title("Objetos Anómalos en Rojo con Índice y Similitud") # Updated title for clarity
    plt.axis("off")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ Plot de anomalías guardado en: {save_path}")

    plt.show()
    plt.close()
# --- Fin de la definición de la función show_anomalies_on_image ---
# --- Nuevas funciones de ploteo para la matriz P y P_augmented_full ---
def plot_assignment_matrix(P_matrix, query_labels, reference_labels, save_path=None, title="Matriz de Asignación P"):
    """
    Visualiza la matriz de asignación P como un mapa de calor.

    Args:
        P_matrix (torch.Tensor or np.array): La matriz de asignación (M x N).
        query_labels (list): Etiquetas para los objetos de consulta (eje Y).
        reference_labels (list): Etiquetas para los objetos de referencia (eje X).
        save_path (str, optional): Ruta para guardar la imagen del plot.
        title (str): Título del plot.
    """
    if isinstance(P_matrix, torch.Tensor):
        #P_matrix = P_matrix.cpu().numpy()
        P_matrix = P_matrix.detach().cpu().numpy()

    plt.figure(figsize=(P_matrix.shape[1] * 0.8 + 2, P_matrix.shape[0] * 0.8 + 2))
    plt.imshow(P_matrix, cmap='viridis', origin='upper', aspect='auto')
    plt.colorbar(label='Probabilidad de Asignación')
    plt.xticks(np.arange(len(reference_labels)), reference_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(query_labels)), query_labels)
    plt.xlabel('Objetos de Referencia')
    plt.ylabel('Objetos de Consulta')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot de la matriz de asignación guardado en: {save_path}")
    plt.show()
    plt.close()

def plot_augmented_assignment_matrix(P_augmented_full, query_labels, reference_labels, save_path=None, title="Matriz de Asignación Aumentada (con Trash Bin)"):
    """
    Visualiza la matriz de asignación aumentada (incluyendo los trash bins) como un mapa de calor.

    Args:
        P_augmented_full (torch.Tensor or np.array): La matriz de asignación aumentada ((M+1) x (N+1)).
        query_labels (list): Etiquetas para los objetos de consulta.
        reference_labels (list): Etiquetas para los objetos de referencia.
        save_path (str, optional): Ruta para guardar la imagen del plot.
        title (str): Título del plot.
    """
    if isinstance(P_augmented_full, torch.Tensor):
        #P_augmented_full = P_augmented_full.cpu().numpy()
        P_augmented_full = P_augmented_full.detach().cpu().numpy()

    # Añadir etiquetas para los trash bins
    full_query_labels = [f"Q_{i}" for i in query_labels] + ["Trash Bin (Q)"]
    full_reference_labels = [f"R_{i}" for i in reference_labels] + ["Trash Bin (R)"]

    plt.figure(figsize=(P_augmented_full.shape[1] * 0.8 + 2, P_augmented_full.shape[0] * 0.8 + 2))
    plt.imshow(P_augmented_full, cmap='viridis', origin='upper', aspect='auto')
    plt.colorbar(label='Probabilidad de Asignación')
    plt.xticks(np.arange(len(full_reference_labels)), full_reference_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(full_query_labels)), full_query_labels)
    plt.xlabel('Objetos de Referencia y Trash Bin')
    plt.ylabel('Objetos de Consulta y Trash Bin')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot de la matriz de asignación aumentada guardado en: {save_path}")
    plt.show()
    plt.close()


# --- Fin de las nuevas funciones de ploteo ---

## Matching-continue---
## Matching
start_time_sam_matching = time.time()



import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_global_max_pool(feat_map):
    return F.adaptive_max_pool2d(feat_map, output_size=1).squeeze(-1).squeeze(-1)

class SimpleObjectMatchingModule(nn.Module):
    def __init__(self, sinkhorn_iterations=100, sinkhorn_epsilon=0.1, bin_score_value=0.5):
        super(SimpleObjectMatchingModule, self).__init__()
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.z = nn.Parameter(torch.tensor(bin_score_value, dtype=torch.float32))

    def forward(self, d_M_q, d_N_r):
        M = d_M_q.shape[0]
        N = d_N_r.shape[0]

        if M == 0 or N == 0:
            return torch.empty(M, N, device=d_M_q.device), \
                   torch.empty(M+1, N+1, device=d_M_q.device)

        score_matrix = torch.mm(d_M_q, d_N_r.T)
        #print("score_matrix (antes de Sinkhorn):\n", score_matrix)

        S_augmented = torch.zeros((M + 1, N + 1), device=d_M_q.device, dtype=d_M_q.dtype)
        S_augmented[:M, :N] = score_matrix
        S_augmented[:M, N] = self.z
        S_augmented[M, :N] = self.z
        S_augmented[M, N] = self.z
        print("S_augmented antes de Sinkhorn:\n", S_augmented)

        K = torch.exp(S_augmented / self.sinkhorn_epsilon)
        print("K (antes de Sinkhorn):\n", K)
        

        for i in range(self.sinkhorn_iterations):
            K = K / K.sum(dim=1, keepdim=True)
            K = K / K.sum(dim=0, keepdim=True)
            #print(f"Iteración {i+1}: K.shape = {K}")

        P_augmented_full = K
        P = P_augmented_full[:M, :N]

        return P, P_augmented_full

fobj_q_pooled = apply_global_max_pool(fobj_q)
print("Shape de fobj_q_pooled:", fobj_q_pooled.shape)
print("Máximo de fobj_q_pooled:", torch.max(fobj_q_pooled).item())
print("Mínimo de fobj_q_pooled:", torch.min(fobj_q_pooled).item())

all_fobj_r_pooled_list = []
for fobj_r_current in all_fobj_r_list:
    pooled_r = apply_global_max_pool(fobj_r_current)
    all_fobj_r_pooled_list.append(pooled_r)
    
d_M_q = F.normalize(fobj_q_pooled, p=2, dim=1) #shape (M, C)
d_N_r_list = [F.normalize(fobj_r_pooled, p=2, dim=1) 
                              for fobj_r_pooled in all_fobj_r_pooled_list]
print("Máximo de d_M_q:", torch.max(d_M_q).item())
print("Mínimo de d_M_q:", torch.min(d_M_q).item())

object_matching_module = SimpleObjectMatchingModule(
    sinkhorn_iterations=50,
    sinkhorn_epsilon=0.1,
    bin_score_value=2.36
).to(device)

P_matrices = []
P_augmented_full_matrices = []

for i, d_N_r_current_image in enumerate(d_N_r_list):
    d_M_q_cuda = d_M_q.to(device)
    d_N_r_current_image_cuda = d_N_r_current_image.to(device)

    P_current, P_augmented_current = object_matching_module(d_M_q_cuda, d_N_r_current_image_cuda)
    P_matrices.append(P_current)
    P_augmented_full_matrices.append(P_augmented_current)


print("\n--- Matrices P y P_augmented_full generadas ---")
# --- NUEVOS DICCIONARIOS CONSOLIDADOS ---
# Almacenarán para cada query_idx, las referencias que le corresponden de TODOS los vecinos.
M = d_M_q.shape[0]
all_matched_ref_indices_by_query_obj = {q_idx: [] for q_idx in range(M)} # M es el número de objetos de consulta (Iq)
all_closest_unmatched_ref_indices_by_query_obj = {q_idx: [] for q_idx in range(M)}
# Imprimir shapes de los diccionarios consolidados
#//////
print("\n--- Resultados Consolidados ---")
print("all_matched_ref_indices_by_query_obj:")
for q_idx, matches in all_matched_ref_indices_by_query_obj.items():
    print(f"  Objeto de Consulta {q_idx}: {matches}")

print("\nall_closest_unmatched_ref_indices_by_query_obj:")
for q_idx, closest_unmatches in all_closest_unmatched_ref_indices_by_query_obj.items():
    print(f"  Objeto de Consulta {q_idx}: {closest_unmatches}")
#/////////////////
# Procesar matrices P y P_augmented_full para obtener índices
for i, (P, P_augmented_full) in enumerate(zip(P_matrices, P_augmented_full_matrices)):
    current_neighbor_key = f"Vecino_{i+1}"
    N_current = P.shape[1] 

    print(f"\n--- Vecino {current_neighbor_key} ---")
    print(f"Matriz P (MxN) para el vecino {current_neighbor_key}:")
    print(P)
    print(f"Matriz P_augmented_full (M+1 x N+1) para el vecino {current_neighbor_key}:")
    print(P_augmented_full)

    # Imprimir sumas de filas y columnas de P_augmented_full
    augmented_with_totals = torch.cat([
        torch.cat([P_augmented_full, P_augmented_full.sum(dim=0, keepdim=True)], dim=0),
        torch.cat([P_augmented_full.sum(dim=1, keepdim=True), P_augmented_full.sum().unsqueeze(0).unsqueeze(0)], dim=0)
    ], dim=1)
    print(f"Matriz P_augmented_full con totales (M+2 x N+2):\n{augmented_with_totals}")

    print(f"\n--- Decisiones de Emparejamiento para el Vecino {current_neighbor_key} ---")
    for obj_idx in range(P.shape[0]):
        
        # Obtener la probabilidad más alta dentro de P y su índice
        if N_current > 0:
            max_prob_P = P[obj_idx].max().item()
            max_idx_P = P[obj_idx].argmax().item()
        else:
            max_prob_P = -float('inf')
            max_idx_P = -1

        # Encontrar el índice del segundo objeto de referencia más similar en P
        second_max_idx_P = -1

        if N_current >= 2:
            temp_P_row = P[obj_idx].clone()
            temp_P_row[max_idx_P] = -float('inf')
            
            second_max_prob_P = temp_P_row.max().item()
            second_max_idx_P = temp_P_row.argmax().item()
        elif N_current == 1:
             second_max_prob_P = -float('inf')
             second_max_idx_P = -1
        else:
             second_max_prob_P = -float('inf')
             second_max_idx_P = -1

        trash_bin_prob = P_augmented_full[obj_idx, -1].item() 

        print(f"  Objeto de Consulta {obj_idx}:")
        print(f"    Probabilidad máxima en P: {max_prob_P:.4f} en el índice {max_idx_P}")
        if N_current >= 2:
            print(f"    Segunda prob. máxima en P: {second_max_prob_P:.4f} en el índice {second_max_idx_P}")
        else:
            print(f"    (Menos de 2 objetos de referencia, no hay 'segunda prob. máxima' válida)")

        print(f"    Probabilidad en el 'Trash Bin': {trash_bin_prob:.4f}")

        # Decisión y almacenamiento en los diccionarios consolidados
        if trash_bin_prob > max_prob_P:
            # No emparejado: añadir el "casi-par" (segundo más similar) a la lista de ese objeto de consulta
            if second_max_idx_P != -1: # Solo añadir si hay un "casi-par" válido
                all_closest_unmatched_ref_indices_by_query_obj[obj_idx].append((i, second_max_idx_P)) # (índice_vecino, índice_referencia)
            print(f"    Decisión: DESEMPAREJADO. 'Casi-par' (2do más similar en P): objeto {second_max_idx_P}")
        else:
            # Emparejado: añadir el emparejamiento real a la lista de ese objeto de consulta
            all_matched_ref_indices_by_query_obj[obj_idx].append((i, max_idx_P)) # (índice_vecino, índice_referencia)
            print(f"    Decisión: EMPAREJADO con objeto de imagen {max_idx_P}")

# --- Resultados Finales Consolidados ---
print("\n--- Resultados Finales Consolidados (Índices) ---")
print("all_matched_ref_indices_by_query_obj (query_idx: [(vecino_idx, ref_idx), ...]):")
for q_idx, matches in all_matched_ref_indices_by_query_obj.items():
    print(f"  Query {q_idx}: {matches}")

print("\nall_closest_unmatched_ref_indices_by_query_obj (query_idx: [(vecino_idx, second_ref_idx), ...]):")
for q_idx, closest_unmatches in all_closest_unmatched_ref_indices_by_query_obj.items():
    print(f"  Query {q_idx}: {closest_unmatches}")

# --- EJEMPLO: Cómo usar los resultados consolidados para extraer los mapas de características ---
print("\n--- EJEMPLO: Cómo extraer mapas de características usando los resultados consolidados ---")


# Aplicar para todos los objetos de consulta disponibles
for query_to_check in range(len(fobj_q)):
    print(f"\n--- Extrayendo para Objeto de Consulta {query_to_check} (Emparejados) ---")
    for neighbor_idx, ref_idx in all_matched_ref_indices_by_query_obj[query_to_check]:
        query_fmap = fobj_q[query_to_check]  # Mapa de características del objeto de consulta
        # Mapa de características del objeto de referencia emparejado del vecino específico
        ref_fmap = all_fobj_r_list[neighbor_idx][ref_idx]
        print(f"  Query M {query_to_check} (shape: {query_fmap.shape}) se emparejó con ref N {ref_idx} del Vecino {neighbor_idx+1} (shape: {ref_fmap.shape})")

    print(f"\n--- Extrayendo para Objeto de Consulta {query_to_check} (No Emparejados - Casi-Par) ---")
    for neighbor_idx, second_ref_idx in all_closest_unmatched_ref_indices_by_query_obj[query_to_check]:
        query_fmap = fobj_q[query_to_check]
        # Mapa de características del "casi-par" de referencia del vecino específico
        ref_fmap_closest = all_fobj_r_list[neighbor_idx][second_ref_idx]
        print(f"  Query M {query_to_check} (shape: {query_fmap.shape}) fue no emparejado, su 'casi-par' es ref N {second_ref_idx} del Vecino {neighbor_idx+1} (shape: {ref_fmap_closest.shape})")







exit()
# Ejemplo para el Objeto de Consulta 0:
query_to_check = 2

print(f"\n--- Extrayendo para Objeto de Consulta {query_to_check} (Emparejados) ---")
for neighbor_idx, ref_idx in all_matched_ref_indices_by_query_obj[query_to_check]:
    query_fmap = fobj_q[query_to_check] # Mapa de características del objeto de consulta
    # Mapa de características del objeto de referencia emparejado del vecino específico
    ref_fmap = all_fobj_r_list[neighbor_idx][ref_idx] 
    print(f"  Query M {query_to_check} (shape: {query_fmap.shape}) se emparejó con ref N {ref_idx} del Vecino {neighbor_idx+1} (shape: {ref_fmap.shape})")

print(f"\n--- Extrayendo para Objeto de Consulta {query_to_check} (No Emparejados - Casi-Par) ---")
for neighbor_idx, second_ref_idx in all_closest_unmatched_ref_indices_by_query_obj[query_to_check]:
    query_fmap = fobj_q[query_to_check]
    # Mapa de características del "casi-par" de referencia del vecino específico
    ref_fmap_closest = all_fobj_r_list[neighbor_idx][second_ref_idx]
    print(f"  Query {query_to_check} (shape: {query_fmap.shape}) fue no emparejado, su 'casi-par' es ref {second_ref_idx} del Vecino {neighbor_idx+1} (shape: {ref_fmap_closest.shape})")









exit()
di_Matched = {}
di_Unmatched = {}
# Imprimir las matrices P y P_augmented_full
for i, (P, P_augmented_full) in enumerate(zip(P_matrices, P_augmented_full_matrices)):
    print(f"\n--- Vecino {i + 1} ---")
    print(f"Matriz P (MxN) para el vecino {i + 1}:")
    print(P)
    print(f"Matriz P_augmented_full (M+1 x N+1) para el vecino {i + 1}:")
    print(P_augmented_full)

    # Imprimir la suma de las filas y columnas de P_augmented_full
    augmented_with_totals = torch.cat([
        torch.cat([P_augmented_full, P_augmented_full.sum(dim=0, keepdim=True)], dim=0),
        torch.cat([P_augmented_full.sum(dim=1, keepdim=True), P_augmented_full.sum().unsqueeze(0).unsqueeze(0)], dim=0)
    ], dim=1)
    
    print(f"Matriz P_augmented_full con totales (M+2 x N+2):\n{augmented_with_totals}")

    matched_results = [] # To store (query_idx, matched_ref_idx)
    unmatched_results = [] # To store (query_idx, closest_ref_idx_in_P)

    print(f"\n--- Decisiones de Emparejamiento para el Vecino {i + 1} ---")
    # Iterar sobre cada objeto de consulta
    for obj_idx in range(P.shape[0]):
        max_prob_P = P[obj_idx].max()
        max_idx_P = P[obj_idx].argmax() # Index of the best match in P

        # This is the "closest matching reference object descriptor vector" 
        # for unmatched objects, as per your clarification.
        closest_ref_idx_in_P = max_idx_P.item() 
        
        trash_bin_prob = P_augmented_full[obj_idx, -1] # Probability of being unmatched (trash bin)

        print(f"  Objeto de Consulta {obj_idx}:")
        print(f"    Probabilidad máxima en P (emparejamiento con objeto de imagen): {max_prob_P:.4f} en el índice {closest_ref_idx_in_P}")
        print(f"    Probabilidad en el 'Trash Bin' (P_augmented_full): {trash_bin_prob:.4f}")

        # Verificar si la probabilidad más alta está en el trash bin
        if trash_bin_prob > max_prob_P:
            unmatched_results.append((obj_idx, closest_ref_idx_in_P))
            print(f"    Decisión: DESEMPAREJADO (Prob. 'Trash Bin' {trash_bin_prob:.4f} > Prob. Máx en P {max_prob_P:.4f}). Más cercano en P: objeto {closest_ref_idx_in_P}")
        else:
            matched_results.append((obj_idx, closest_ref_idx_in_P))
            print(f"    Decisión: EMPAREJADO (Prob. Máx en P {max_prob_P:.4f} >= Prob. 'Trash Bin' {trash_bin_prob:.4f}) con objeto de imagen {closest_ref_idx_in_P}")

    # Convertir a tensores
        # di_Matched will contain (query_idx, matched_ref_idx)
        di_Matched[f"Vecino_{i+1}"] = torch.tensor(matched_results, dtype=torch.long, device=P.device)
        # di_Unmatched will contain (query_idx, closest_ref_idx_in_P)
        di_Unmatched[f"Vecino_{i+1}"] = torch.tensor(unmatched_results, dtype=torch.long, device=P.device)

        print("\n--- Resultados Finales ---")
        print("di_Matched:", di_Matched)
        print("di_Unmatched:", di_Unmatched)



exit()
di_Matched = {}
di_Unmatched = {}
# Imprimir las matrices P y P_augmented_full
for i, (P, P_augmented_full) in enumerate(zip(P_matrices, P_augmented_full_matrices)):
    print(f"\n--- Vecino {i + 1} ---")
    print(f"Matriz P (MxN) para el vecino {i + 1}:")
    print(P)
    print(f"Matriz P_augmented_full (M+1 x N+1) para el vecino {i + 1}:")
    print(P_augmented_full)

# Imprimir la suma de las filas y columnas de P_augmented_full
    augmented_with_totals = torch.cat([
        torch.cat([P_augmented_full, P_augmented_full.sum(dim=0, keepdim=True)], dim=0),
        torch.cat([P_augmented_full.sum(dim=1, keepdim=True), P_augmented_full.sum().unsqueeze(0).unsqueeze(0)], dim=0)
    ], dim=1)
    
    print(f"Matriz P_augmented_full con totales (M+2 x N+2):\n{augmented_with_totals}")

    matched_indices = []
    unmatched_indices = []

    # Iterar sobre cada objeto de consulta
    for obj_idx in range(P.shape[0]):
        max_prob = P[obj_idx].max()
        max_idx = P[obj_idx].argmax()

        # Verificar si la probabilidad más alta está en el trash bin
        if P_augmented_full[obj_idx, -1] > P_augmented_full[obj_idx, max_idx]:
            unmatched_indices.append(obj_idx)
        else:
            matched_indices.append((obj_idx, max_idx.item()))

    # Convertir a tensores
    di_Matched[f"Vecino_{i+1}"] = torch.tensor(matched_indices, dtype=torch.long, device=P.device)
    di_Unmatched[f"Vecino_{i+1}"] = torch.tensor(unmatched_indices, dtype=torch.long, device=P.device)

print("di_Matched:", di_Matched)
print("di_Unmatched:", di_Unmatched)







exit(   )
def apply_global_max_pool(feat_map):
    return F.adaptive_max_pool2d(feat_map, output_size=1).squeeze(-1).squeeze(-1)

fobj_q_pooled = apply_global_max_pool(fobj_q)
print("Shape de fobj_q_pooled:", fobj_q_pooled.shape)
print("Máximo de fobj_q_pooled:", torch.max(fobj_q_pooled).item())
print("Mínimo de fobj_q_pooled:", torch.min(fobj_q_pooled).item())

all_fobj_r_pooled_list = []
for fobj_r_current in all_fobj_r_list:
    pooled_r = apply_global_max_pool(fobj_r_current) # tensor (N, C)
    all_fobj_r_pooled_list.append(pooled_r)          # # --> Lista de tensores (N, C)
    
d_M_q = F.normalize(fobj_q_pooled, p=2, dim=1) # --> d_M_q tensor (M, C)
d_N_r_list = [F.normalize(fobj_r_pooled, p=2, dim=1)  # 
                          for fobj_r_pooled in all_fobj_r_pooled_list]# --> d_N_r tensor (k, N, C)
print("Máximo de d_M_q:", torch.max(d_M_q).item())
print("Mínimo de d_M_q:", torch.min(d_M_q).item())



import torch
from torch import nn
from typing import List, Tuple

# --- Funciones necesarias copiadas del código que proporcionaste ---

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Realiza la normalización de Sinkhorn en espacio logarítmico para estabilidad. """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        Z = Z + u.unsqueeze(2) + v.unsqueeze(1) # Actualiza Z dentro del bucle
        # AÑADE ESTA LÍNEA PARA VER LA MATRIZ EN CADA ITERACIÓN
        print(f"--- Iteración {i+1} de Sinkhorn ---")
        print("Z_actual (en log):", Z)
    return Z #+ u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Realiza el Transporte Óptimo Diferenciable en espacio logarítmico. """
    b, m, n = scores.shape
    one = scores.new_tensor(1) # Crea un tensor en el mismo dispositivo que scores
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    # Añadir filas y columnas para el 'trash bin'
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha_single = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha_single], -1)], 1)

    norm1 = - (ms + ns).log()
    log_mu = torch.cat([norm1.expand(m), ns.log()[None] + norm1])
    log_nu = torch.cat([norm1.expand(n), ms.log()[None] + norm1])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    # Ejecutar las iteraciones de Sinkhorn
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm1  # Multiplicar probabilidades por M+N
    return Z

# --- Preparación de tus datos de descriptores (ejemplo con datos dummy) ---

# Determinar el dispositivo a usar (GPU si está disponible, si no CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

num_query_desc = 50   # M
num_ref_desc = 70     # N
descriptor_dim = 256  # C

# Tus descriptores (usando datos aleatorios para la demostración)
# Asegurarse de que los tensores iniciales estén en el dispositivo correcto
# Si ya tienes d_M_q y d_N_r_list definidos, asegúrate de moverlos al 'device'
d_M_q = d_M_q.to(device)
d_N_r = d_N_r_list[0].to(device)
print("Primeros 5 valores de d_M_q:")
print(d_M_q[:5])

print("Primeros 5 valores de d_N_r:")
print(d_N_r[:5])
# Imprimir el máximo y el mínimo de d_M_q y d_N_r
print("Máximo de d_M_q:", torch.max(d_M_q).item())
print("Mínimo de d_M_q:", torch.min(d_M_q).item())

print("Máximo de d_N_r:", torch.max(d_N_r).item())
print("Mínimo de d_N_r:", torch.min(d_N_r).item())
#d_M_q = torch.randn(num_query_desc, descriptor_dim).to(device) # Ejemplo de tus descriptores de consulta
#d_N_r = torch.randn(num_ref_desc, descriptor_dim).to(device)   # Ejemplo de tus descriptores de referencia

print("Shape de d_M_q:", d_M_q.shape)
print("Shape de d_N_r:", d_N_r.shape)
print("Tipo de d_N_r:", type(d_N_r))

# Para que coincida con el formato de SuperGlue (Batch, Dim, N_keypoints) para einsum
# d_M_q (M, C) -> mdesc0 (1, C, M)
# d_N_r (N, C) -> mdesc1 (1, C, N)
mdesc0_superglue_format = d_M_q.permute(1, 0).unsqueeze(0)
mdesc1_superglue_format = d_N_r.permute(1, 0).unsqueeze(0)
print("Shape de mdesc0_superglue_format:", mdesc0_superglue_format.shape)
print("Shape de mdesc1_superglue_format:", mdesc1_superglue_format.shape)

# --- Cálculo de la matriz de puntuación S_ij ---
# Esto es equivalente a 'scores' en el código de SuperGlue
scores = torch.einsum('bdn,bdm->bnm', mdesc0_superglue_format, mdesc1_superglue_format)
print("Primeros 20 valores de scores:")
print(scores.flatten()[:20])
print("Tipo de scores:", type(scores))
print("Shape de scores:", scores.shape)
print("Máximo de scores:", torch.max(scores).item())
print("Mínimo de scores:", torch.min(scores).item())

#scores = scores / d_M_q.shape[1]**0.5  # Normalización como en SuperGlue
print("d_M_q.shape[1]:", d_M_q.shape[1])
# --- Parámetro 'trash bin' (z) y número de iteraciones de Sinkhorn ---
# Asegurarse de que el parámetro también esté en el dispositivo correcto
trash_bin_param = torch.nn.Parameter(torch.tensor(0.01, device=device))

sinkhorn_iterations = 30 # Número de iteraciones, como en la configuración por defecto de SuperGlue

# --- Aplicar Transporte Óptimo para obtener la matriz P aumentada (en espacio logarítmico) ---
# Z_augmented_log es el resultado de la aplicación de Sinkhorn sobre la matriz de puntuaciones
# aumentada con el 'trash bin'. Es la matriz P aumentada en espacio logarítmico.
log_optimal_scale = 2.36 # Puedes ajustar este valor si lo necesitas
Z_augmented_log = log_optimal_transport(
    torch.log(scores) + log_optimal_scale, # <--- ¡CAMBIO CLAVE AQUÍ!
    torch.tensor(0.0, device=device),      # <--- ¡Y AQUÍ! (log de un score de trash bin de 1.0 es 0.0)
    iters=sinkhorn_iterations)

# --- Recuperar las matrices P ---

# 1. Matriz P aumentada con el trash bin (en espacio normal, no logarítmico)
# Simplemente se toma la exponencial del resultado de log_optimal_transport
P_augmented = torch.exp(Z_augmented_log)

# 2. Matriz P original (sin el trash bin, en espacio normal)
# Esto se obtiene tomando la porción de la matriz aumentada antes de la última fila y columna.
P_original = torch.exp(Z_augmented_log[:, :-1, :-1])

# --- Imprimir resultados ---
print(f"Forma de la matriz de puntuación (S_ij): {scores.shape}")
print(f"Forma de la matriz P aumentada (en log): {Z_augmented_log.shape}")
print(f"Forma de la matriz P aumentada (normal): {P_augmented.shape}")
print(f"Forma de la matriz P original (sin trash bin, normal): {P_original.shape}")

# Imprimir las matrices y sus tipos
print("\n--- Matrices y Tipos ---")
print("Scores Matrix (S_ij):")
print(scores)
print("Tipo de Scores Matrix:", type(scores))

print("\nZ_augmented_log (Matriz P aumentada en log):")
print(Z_augmented_log)
print("Tipo de Z_augmented_log:", type(Z_augmented_log))

print("\nP_augmented (Matriz P aumentada):")
print(P_augmented)
print("Tipo de P_augmented:", type(P_augmented))

print("\nP_original (Matriz P original):")
print(P_original)
print("Tipo de P_original:", type(P_original))




# Asumiendo que Z_augmented_log es el resultado de log_optimal_transport
# y tiene forma (B, M_aug, N_aug) donde M_aug = M+1, N_aug = N+1

# Extraer solo la parte relevante para los descriptores originales
scores_original_space = Z_augmented_log[:, :-1, :-1].exp() # Es tu P_original

# Obtener el índice del máximo por cada fila (para descriptores de consulta)
# y por cada columna (para descriptores de referencia)
max0, max1 = scores_original_space.max(2), scores_original_space.max(1)
indices0, indices1 = max0.indices, max1.indices

# Crear rangos de índices para la verificación de exclusividad mutua
# (similar a arange_like en SuperGlue)
arange_like = lambda x, dim: x.new_ones(x.shape[dim]).cumsum(0) - 1

# Verificar la exclusividad mutua
mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)

# Aplicar umbral de coincidencia y exclusividad mutua
match_threshold = 0.2 # Puedes ajustar este valor si lo necesitas

# Inicializar scores de coincidencia y asignaciones a cero/sin match
zero = scores_original_space.new_tensor(0)
mscores0 = torch.where(mutual0, max0.values, zero) # Aquí usamos max0.values directamente del log-space Z_augmented_log
mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

# Filtrar por umbral de confianza
valid0 = mutual0 & (mscores0 > match_threshold) # mscores0 ya está en espacio normal si usas .exp()
valid1 = mutual1 & valid0.gather(1, indices1)

# Crear las asignaciones finales
# -1 indica que no hay coincidencia
matches0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
matches1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

print("\n--- Asignaciones Duras ---")
print("Matches0 (para descriptores de consulta):", matches0)
print("Matches1 (para descriptores de referencia):", matches1)
print("Matching Scores0 (para descriptores de consulta):", mscores0) # Estas son las puntuaciones de confianza de las coincidencias



exit()






# Comparar d_M_q con cada d_N_r
M, C = d_M_q.shape
k = len(d_N_r_list)
N_total = sum([d.shape[0] for d in d_N_r_list])  # suma de todos los N
# Imprimir el valor de k y N_total
print("Número de vecinos (k):", k)
print("Número total de objetos de referencia (N_total):", N_total)

# Concatenamos todos los descriptores de referencia (k*N, C)
d_N_full = torch.cat(d_N_r_list, dim=0)  # (N_total, C)

# Creamos matriz de similitud coseno (M, N_total)
S_ij = torch.matmul(d_M_q, d_N_full.T)  # producto escalar ya es coseno si están normalizados


print("Matriz de similitud S_ij:", S_ij)  # (M x N_total)
# Imprimir el tipo de S_ij
print("Tipo de S_ij:", type(S_ij))
# Imprimir los 5 primeros valores de S_ij
if S_ij.numel() > 0:
    print("Primeros 5 valores de S_ij:")
    print(S_ij.flatten()[:5])
else:
    print("S_ij está vacío.")




# Imprimir el tipo de d_M_q y d_N_r_list
print("Tipo de d_M_q:", type(d_M_q))
print("Tipo de d_N_r_list:", type(d_N_r_list))
# Imprimir shape de d_M_q
print("Shape de d_M_q:", d_M_q.shape)

# Imprimir shape de cada tensor en d_N_r_list
for idx, d_N_r in enumerate(d_N_r_list):
    print(f"Shape de d_N_r para vecino {idx + 1}:", d_N_r.shape)

### SCORES MATRIX










exit()


def max_similarities(query_feats, candidate_feats):
    sim_matrix = torch.mm(query_feats, candidate_feats.T)# [-1, 1] producto punto...similar a similitud coseno si se normalizan A y B a vectores unitarios
    max_vals, _ = sim_matrix.max(dim=1)
    return max_vals



# --- Optimal Matching Module ---
class ObjectMatchingModule(nn.Module):
    def __init__(self, superglue_weights_path=None, sinkhorn_iterations=100, sinkhorn_epsilon=0.1):
        super(ObjectMatchingModule, self).__init__()
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon

        if superglue_weights_path and os.path.exists(superglue_weights_path):
            try:
                state_dict = torch.load(superglue_weights_path, map_location=device)
                if 'bin_score' in state_dict:
                    z_value = state_dict['bin_score'].item()
                elif 'match_model.bin_score' in state_dict:
                    z_value = state_dict['match_model.bin_score'].item()
                else:
                    print(f"Advertencia: 'z' (bin_score) no encontrado en {superglue_weights_path}. Inicializando con valor predeterminado.")
                    z_value = 0.5
                                                #aqui puedes variar 0.5 por ej.
                self.z = nn.Parameter(torch.tensor(z_value, dtype=torch.float32))
                print(f"Parámetro 'z' cargado de SuperGlue: {self.z.item():.4f}")

            except Exception as e:
                print(f"Error al cargar 'z' de {superglue_weights_path}: {e}")
                self.z = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            print(f"Advertencia: superglue_weights_path no válido o no encontrado: {superglue_weights_path}. Inicializando 'z' con valor predeterminado.")
            self.z = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, d_M_q, d_N_r):
        M = d_M_q.shape[0]
        N = d_N_r.shape[0]

        if M == 0 or N == 0:
            print(f"Matrices de entrada vacías (M={M}, N={N}). Devolviendo matrices de coincidencia vacías.")
            return torch.empty(M, N, device=d_M_q.device), \
                   torch.empty(M+1, N+1, device=d_M_q.device)

        score_matrix = torch.mm(d_M_q, d_N_r.T)
        print(f"Matriz de similitud inicial (MxN): {score_matrix.shape}")
        # print("score_matrix:\n", score_matrix) # Descomentar para ver la matriz

        S_augmented = torch.zeros((M + 1, N + 1), device=d_M_q.device, dtype=d_M_q.dtype)
        S_augmented[:M, :N] = score_matrix
        S_augmented[:M, N] = self.z # Última columna (trash bin para query)
        S_augmented[M, :N] = self.z # Última fila (trash bin para reference)
        S_augmented[M, N] = self.z # Esquina inferior derecha (trash bin vs trash bin)
        print(f"Matriz S_augmented (M+1 x N+1): {S_augmented.shape}")
        # print("S_augmented:\n", S_augmented) # Descomentar para ver la matriz

        K = torch.exp(S_augmented / self.sinkhorn_epsilon)
        # print("K (exp(S/epsilon)) antes de Sinkhorn:\n", K) # Descomentar para ver

        for i in range(self.sinkhorn_iterations):
            K = K / K.sum(dim=1, keepdim=True) # Normalizar filas
            K = K / K.sum(dim=0, keepdim=True) # Normalizar columnas
            # print(f"K después de iteración {i+1} de Sinkhorn:\n", K) # Descomentar para ver cada iteración

        P_augmented_full = K
        P = P_augmented_full[:M, :N] # La matriz de asignación sin trash bins
        print(f"Matriz de asignación P (MxN): {P.shape}")
        print("P:\n", P) # Print de la matriz de asignación P
        print(f"Matriz de asignación P_augmented_full (M+1 x N+1): {P_augmented_full.shape}")
        print("P_augmented_full:\n", P_augmented_full) # Print de la matriz de asignación completa

        return P, P_augmented_full

# --- Uso del módulo de coincidencia ---
superglue_weights_path = "/home/imercatoma/superglue_indoor.pth"

object_matching_module = ObjectMatchingModule(
    superglue_weights_path=superglue_weights_path,
    sinkhorn_iterations=100,
    sinkhorn_epsilon=0.1
).to(device)








# DATOS PARA EL AMM
query_lr_features = query_lr_features   # shape (1, C, 8H', 8W')
query_hr_feats = query_hr_feats         # shape (1, C, 8H', 8W')
fobj_q = fobj_q                         # shape (M, C, 8H', 8W')
fobj_r = all_fobj_r_list                #shape (N, C, 8H', 8W') for each neighbor k

# Imprimir el tipo de las variables solicitadas
print("Tipo de query_lr_features:", type(query_lr_features))
print("Tipo de query_hr_feats:", type(query_hr_feats))
print("Tipo de fobj_q:", type(fobj_q))
print("Tipo de fobj_r:", type(fobj_r))



d_M_q = d_M_q                   # shape (M, C)
d_N_r_list = d_N_r_list         # shape (N, C) for each neighbor k

# Imprimir el tipo de d_M_q y d_N_r_list
print("Tipo de d_M_q:", type(d_M_q))
print("Tipo de d_N_r_list:", type(d_N_r_list))

# Imprimir shape, primeros 5 valores, máximo y mínimo de d_M_q
print("Shape de d_M_q:", d_M_q.shape)
if d_M_q.numel() > 0:
    print("Primeros 5 valores de d_M_q:")
    print(d_M_q[:5])
    print("Máximo de d_M_q:", torch.max(d_M_q).item())
    print("Mínimo de d_M_q:", torch.min(d_M_q).item())
else:
    print("d_M_q está vacío.")

# Imprimir shape, primeros 5 valores, máximo y mínimo de d_N_r_list
for idx, d_N_r in enumerate(d_N_r_list):
    print(f"\nVecino {idx + 1}:")
    print("Shape de d_N_r:", d_N_r.shape)
    if d_N_r.numel() > 0:
        print("Primeros 5 valores de d_N_r:")
        print(d_N_r[:5])
        print("Máximo de d_N_r:", torch.max(d_N_r).item())
        print("Mínimo de d_N_r:", torch.min(d_N_r).item())
    else:
        print("d_N_r está vacío.")



# P = 
# p_augmented =

# indice = [objeto, vecino] = [i, j]  # i es el índice del objeto en fobj_q, j es el índice del vecino en fobj_r
# indice_bin = [i, j]  # objeto con el vecino j mas parecido





