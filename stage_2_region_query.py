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
query_image_path = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/test/cut/000.png'
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
nn_finder = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(coreset_features_cpu)
print("NearestNeighbors finder inicializado.")

# --- Cargar Modelo DINOv2 ---
print("Cargando modelo DINOv2...")
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
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

    query_patches_flat = features_lr.squeeze(0).permute(1, 2, 0).reshape(-1, features_lr.shape[1])
    query_patches_flat_cpu = query_patches_flat.cpu().numpy()

    distances_to_nn, _ = nn_finder_instance.kneighbors(query_patches_flat_cpu)
    patch_anomaly_scores = distances_to_nn.flatten()
    sorted_patch_anomaly_scores = np.sort(patch_anomaly_scores)[::-1]

    anomaly_map_lr = patch_anomaly_scores.reshape(H_prime, W_prime)
    anomaly_map_lr_tensor = torch.from_numpy(anomaly_map_lr).unsqueeze(0).unsqueeze(0).to(device)
    anomaly_map_upsampled = F.interpolate(anomaly_map_lr_tensor, size=(input_size, input_size), mode='bilinear', align_corners=False)
    anomaly_map_upsampled = anomaly_map_upsampled.squeeze().cpu().numpy()
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

if query_lr_features is None:
    print(f"Error al obtener características para {base_image_name}. No se puede continuar.")
    exit()

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


### Aplicando Máscaras SAM (para Imágenes Anómalas)


# --- APLICANDO SAM MASK (SOLO SI SE CLASIFICA COMO ANÓMALA Y SAM ESTÁ CARGADO) ---

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