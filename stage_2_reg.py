import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

# Import FeatUp utilities for normalization/denormalization
from featup.util import norm, unnorm

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from scipy.stats import median_abs_deviation

# For anomaly region detection and visualization
from skimage import measure
import matplotlib.patches as patches

# Importamos SAM2AutomaticMaskGenerator y utilidades para SAM
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2

# --- Configuración ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224 # Tamaño de entrada para DINOv2
BACKBONE_PATCH_SIZE = 14 # Tamaño de parche para DINOv2 ViT-S/14
use_norm = True

# Dimensiones espaciales de los mapas de características de baja resolución (H', W')
H_prime = input_size // BACKBONE_PATCH_SIZE
W_prime = input_size // BACKBONE_PATCH_SIZE

# Directorios
TRAIN_GOOD_DIR = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/train/good'
TEST_CRACK_DIR = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/test/crack'
PLOT_SAVE_ROOT_DIR = '/home/imercatoma/FeatUp/plots_anomaly_distances'
os.makedirs(PLOT_SAVE_ROOT_DIR, exist_ok=True)
print(f"Directorio raíz de guardado de plots creado/verificado: '{PLOT_SAVE_ROOT_DIR}'")

# Carpeta para HEATMAPS INDIVIDUALES
HEATMAPS_SAVE_DIR = os.path.join(PLOT_SAVE_ROOT_DIR, 'individual_heatmaps')
os.makedirs(HEATMAPS_SAVE_DIR, exist_ok=True)
print(f"Directorio de guardado de heatmaps individuales creado/verificado: '{HEATMAPS_SAVE_DIR}'")

# Carpeta para REGIONES ANÓMALAS DETECTADAS
ANOMALY_REGIONS_SAVE_DIR = os.path.join(PLOT_SAVE_ROOT_DIR, 'detected_anomaly_regions')
os.makedirs(ANOMALY_REGIONS_SAVE_DIR, exist_ok=True)
print(f"Directorio de guardado de regiones anómalas creado/verificado: '{ANOMALY_REGIONS_SAVE_DIR}'")

# Rutas de archivo para cargar los mapas de características del Coreset
core_bank_filenames_file = os.path.join(TRAIN_GOOD_DIR, 'core_bank_filenames.pt')
coreset_relevant_flat_features_bank_file = os.path.join(TRAIN_GOOD_DIR, 'coreset_relevant_flat_features_bank.pt')
template_features_bank_coreset_file = os.path.join(TRAIN_GOOD_DIR, 'template_features_bank_coreset.pt')

# --- Cargar Datos del Coreset (Matriz 'M' para KNN) ---
print("Cargando datos del coreset relevante y banco de características (M)...")
coreset_relevant_filenames = []
coreset_relevant_flat_features_bank = None
coreset_features = None

try:
    coreset_relevant_filenames = torch.load(core_bank_filenames_file)
    coreset_relevant_flat_features_bank = torch.load(coreset_relevant_flat_features_bank_file).to(device)
    coreset_features = torch.load(template_features_bank_coreset_file).to(device)
    
    print(f"Coreset de características (M) cargado. Dimensión: {coreset_features.shape}")
    print(f"Banco de características plano relevante del coreset cargado. Dimensión: {coreset_relevant_flat_features_bank.shape}")
    print(f"Número de nombres de archivo relevantes cargados: {len(coreset_relevant_filenames)}")

except FileNotFoundError as e:
    print(f"Error al cargar archivos del coreset: {e}. Asegúrate de que la Etapa 1 se ejecutó y los archivos existen.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al cargar o procesar los archivos del coreset: {e}")
    exit()

# Mover el coreset a CPU para sklearn's NearestNeighbors
coreset_features_cpu = coreset_features.cpu().numpy()
print(f"NearestNeighbors finder inicializado con características del coreset.")

# Inicializar NearestNeighbors finder una sola vez (para el coreset)
nn_finder = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(coreset_features_cpu)
print("NearestNeighbors finder inicializado con características del coreset.")

# --- Cargar Modelo DINOv2 ---
print("Cargando modelo DINOv2 para extracción de características...")
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

# Placeholder para plot_feats
def plot_feats(image, lr_feats, hr_feats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(lr_feats.mean(dim=0).cpu().numpy(), cmap='viridis')
    axes[1].set_title('LR Features Mean')
    axes[1].axis('off')

    axes[2].imshow(hr_feats.mean(dim=0).cpu().numpy(), cmap='viridis')
    axes[2].set_title('HR Features Mean')
    axes[2].axis('off')
    plt.tight_layout()

# --- Función Principal para Obtener Puntuaciones de Anomalía para una Imagen ---
def get_anomaly_scores_for_image(image_path, model, image_transform, nn_finder_instance, H_prime, W_prime, device):
    """
    Extrae características DINOv2 para una sola imagen, calcula puntuaciones de anomalía por parche
    basadas en la distancia al coreset.
    Devuelve tanto patch_anomaly_scores (sin ordenar) como sorted_patch_anomaly_scores y la imagen original PIL.
    """
    try:
        query_img_pil = Image.open(image_path).convert("RGB")
        input_tensor = image_transform(query_img_pil).unsqueeze(0).to(device)
    except Exception as e:
        print(f"     Error cargando/transformando imagen {os.path.basename(image_path)}: {e}")
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

# --- Funciones de Cálculo de Métricas ---
def calculate_rms(data):
    return np.sqrt(np.mean(data**2))

def calculate_mad(data):
    return median_abs_deviation(data)

def calculate_median(data):
    return np.median(data)

def calculate_quartile(data, q=25):
    return np.percentile(data, q)

# --- Funciones para los filtros de concentración/extensión y magnitud ---
def calculate_spatial_variance_of_top_patches(patch_anomaly_scores, top_percentage=5.5):
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0:
        return np.nan

    num_patches = patch_anomaly_scores.size
    
    if num_patches != H_prime * W_prime:
        print(f"     Advertencia: Número de parches ({num_patches}) no coincide con H'*W' ({H_prime*W_prime}).")

    num_top = max(1, int(num_patches * top_percentage / 100))

    top_patch_indices = np.argsort(patch_anomaly_scores)[-num_top:]

    row_coords = top_patch_indices // W_prime
    col_coords = top_patch_indices % W_prime

    std_rows = np.std(row_coords) if len(row_coords) > 1 else 0.0
    std_cols = np.std(col_coords) if len(col_coords) > 1 else 0.0

    return std_rows + std_cols

def calculate_active_patches_count_relative_threshold(patch_anomaly_scores, relative_threshold_percentage):
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0:
        return 0

    max_val_in_image = np.max(patch_anomaly_scores)
    
    if max_val_in_image == 0:
        return 0
        
    threshold_val = max_val_in_image * relative_threshold_percentage
    
    active_patches = patch_anomaly_scores[patch_anomaly_scores > threshold_val]
    return len(active_patches)

def calculate_top_percent_average_anomaly(patch_anomaly_scores, top_percent=1):
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0:
        return 0.0
    
    num_patches = patch_anomaly_scores.size
    num_top = max(1, int(num_patches * top_percent / 100))

    sorted_scores = np.sort(patch_anomaly_scores)[::-1]
    top_n_scores = sorted_scores[:num_top]
    
    return np.mean(top_n_scores)

# --- Función para generar y guardar mapas de calor individuales ---
def generate_and_save_heatmap(image_original_pil, anomaly_map_final, sorted_patch_anomaly_scores, save_path, file_idx, image_name_for_title):
    
    num_patches = len(sorted_patch_anomaly_scores)
    num_top_for_q_score = max(1, int(num_patches * 0.01))
    top_anomaly_scores = sorted_patch_anomaly_scores[:num_top_for_q_score]
    q_score = np.mean(top_anomaly_scores)
    
    anomalia_estructural = q_score > 0.27

    print("\n--- Detalles del Mapa de Calor y Q-score ---")
    print("Sorted patch anomaly scores (descending):")
    print(sorted_patch_anomaly_scores[:10])
    print(f"Q-score (promedio del 1% superior de distancias): {q_score:.4f}")
    print(f"Anomalía estructural (umbral > 0.27): {'Sí' if anomalia_estructural else 'No'}")

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

    print("--- Generación del mapa de calor y Q-score completada para esta imagen ---")
    return anomaly_map_final, q_score, anomalia_estructural

# --- BARRIDO POR LOTES Y RECOLECCIÓN DE DATOS ---
image_names_processed = []
rms_mad_distances = []
rms_median_distances = []
rms_q1_distances = []
spatial_variances = []
active_patches_counts = []
top_1_percent_averages = []
image_classifications = []

print(f"\nIniciando procesamiento por lotes de imágenes desde: {TEST_CRACK_DIR}")
try:
    test_image_files = [f for f in os.listdir(TEST_CRACK_DIR) if f.lower().endswith('.png')]
    test_image_files.sort()
    if not test_image_files:
        print(f"No se encontraron imágenes .png en: {TEST_CRACK_DIR}")
except FileNotFoundError:
    print(f"Error: Directorio '{TEST_CRACK_DIR}' no encontrado.")
    test_image_files = []

# --- UMBRAL RELATIVO PARA PARCHES ACTIVOS ---
RELATIVE_ACTIVE_PATCH_THRESHOLD_PERCENTAGE = 0.80

# Load SAM2 model once
checkpoint = "/home/imercatoma/sam2_repo_independent/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_model = None
try:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=True)
    print("SAM2 cargado correctamente para Automatic Mask Generation")
except FileNotFoundError as e:
    print(f"Error: No se encontraron los archivos del modelo SAM2: {e}. SAM no estará disponible.")
except Exception as e:
    print(f"Error al cargar SAM2: {e}. SAM no estará disponible.")

for i, image_filename in enumerate(test_image_files):
    query_image_path = os.path.join(TEST_CRACK_DIR, image_filename)
    base_image_name = os.path.basename(query_image_path)
    print(f"\n--- Procesando imagen {i+1}/{len(test_image_files)} (Query): {base_image_name} ---")

    # Call get_anomaly_scores_for_image for the CURRENT query_image_path
    current_patch_anomaly_scores, current_sorted_patch_anomaly_scores, query_img_pil, anomaly_map_final_for_regions, query_lr_features = get_anomaly_scores_for_image(
        query_image_path, dinov2_model, transform, nn_finder, H_prime, W_prime, device
    )

    if query_lr_features is None:
        print(f"Error al obtener características de LR para la imagen {base_image_name}. Saltando esta imagen.")
        continue

    # --- Generar y guardar heatmap individual para la imagen actual ---
    heatmap_filename = f"heat_{base_image_name}"
    individual_heatmap_save_path = os.path.join(HEATMAPS_SAVE_DIR, heatmap_filename)
    
    current_anomaly_map_final, current_q_score, current_anomalia_estructural = \
        generate_and_save_heatmap(query_img_pil, anomaly_map_final_for_regions,
                                   current_sorted_patch_anomaly_scores, 
                                   individual_heatmap_save_path, i, base_image_name.replace(".png", ""))
    
    # Calculate metrics for the current image
    min_val = np.min(current_sorted_patch_anomaly_scores)
    max_val = np.max(current_sorted_patch_anomaly_scores)
    if max_val == min_val:
        normalized_data = np.zeros_like(current_sorted_patch_anomaly_scores, dtype=float)
    else:
        normalized_data = (current_sorted_patch_anomaly_scores - min_val) / (max_val - min_val)

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

    image_names_processed.append(base_image_name)
    rms_mad_distances.append(dist_rms_mad)
    rms_median_distances.append(dist_rms_median)
    rms_q1_distances.append(dist_rms_q1)
    spatial_variances.append(spatial_var)
    active_patches_counts.append(active_count)
    top_1_percent_averages.append(top_1_avg)

    # Classification logic for the current image
    classification = 0
    if top_1_avg >= 0.30:
        classification = 1
        print(f"    Clasificación: ANOMALÍA GRANDE (Top 1% Avg: {top_1_avg:.4f} >= 0.30)")
    elif 0.17 <= top_1_avg < 0.30:
        print(f"    Clasificación: Entrando en evaluación de anomalía leve/buena (Top 1% Avg: {top_1_avg:.4f})")
        initial_classification_based_on_active_patches = 0
        if active_count > 5:
            initial_classification_based_on_active_patches = 1
            print(f"    -> ANOMALÍA LEVE (Parches Activos: {active_count} > 5)")
            classification = 1
        else:
            print(f"    -> Parches Activos ({active_count}) <= 5. Evaluando condiciones de 'buena'.")
            initial_classification_based_on_active_patches = 0
            if dist_rms_median <= 0.055:
                print(f"    -> Condición Buena II (RMS - Mediana <= 0.055): True ({dist_rms_median:.4f})")
                cond_I_met = spatial_var >= 5.5
                print(f"    -> Condición Buena I (Varianza Espacial >= 5.5): {'True' if cond_I_met else 'False'} ({spatial_var:.2f})")
                cond_III_met = dist_rms_mad >= 0.21
                print(f"    -> Condición Buena III (RMS - MAD >= 0.21): {'True' if cond_III_met else 'False'} ({dist_rms_mad:.4f})")
                if cond_I_met or cond_III_met:
                    classification = 0
                    print(f"    -> IMAGEN BUENA (Condición II True, y al menos una de I o III es True)")
                else:
                    classification = initial_classification_based_on_active_patches
                    print(f"    -> { 'ANOMALÍA LEVE' if classification == 1 else 'IMAGEN BUENA' } (Condición II True, pero I y III False. Revertiendo a clasificación por Parches Activos: { 'ANOMALÍA LEVE' if initial_classification_based_on_active_patches == 1 else 'BUENA' })")
            else:
                classification = 1
                print(f"    -> Condición Buena II (RMS - Mediana <= 0.055): False ({dist_rms_median:.4f})")
                print(f"    -> ANOMALÍA LEVE (Condición II es False, clasificación automática como anomalía leve)")
    image_classifications.append(classification)

    # --- Detección y visualización de regiones de anomalía "fuertes" (solo si la imagen es anómala) ---
    if classification == 1:
        start_time_region_detection = time.time()
        print("\n  ** Clasificada como ANÓMALA. Buscando y visualizando regiones con anomalías fuertes... **")
        strong_anomaly_region_threshold = 0.75
        binary_strong_anomaly_map = anomaly_map_final_for_regions > strong_anomaly_region_threshold
        if not np.any(binary_strong_anomaly_map):
            print(f"    No se encontraron píxeles por encima del umbral de {strong_anomaly_region_threshold} para regiones anómalas fuertes. Reduce el umbral si es necesario.")
        else:
            labeled_anomaly_regions = measure.label(binary_strong_anomaly_map)
            region_properties = measure.regionprops(labeled_anomaly_regions)
            detected_strong_anomaly_regions = []
            min_region_pixel_area = 50
            print(f"    Analizando {len(region_properties)} regiones conectadas antes de filtrar.")
            original_img_width, original_img_height = query_img_pil.size
            scale_x = original_img_width / input_size
            scale_y = original_img_height / input_size
            print(f"    Factores de escala para bbox: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")

            for region in region_properties:
                if region.area >= min_region_pixel_area:
                    min_y_at_input_size, min_x_at_input_size, max_y_at_input_size, max_x_at_input_size = region.bbox
                    scaled_min_x = int(np.clip(min_x_at_input_size * scale_x, 0, original_img_width))
                    scaled_min_y = int(np.clip(min_y_at_input_size * scale_y, 0, original_img_height))
                    scaled_max_x = int(np.clip(max_x_at_input_size * scale_x, 0, original_img_width))
                    scaled_max_y = int(np.clip(max_y_at_input_size * scale_y, 0, original_img_height))
                    region_width = scaled_max_x - scaled_min_x
                    region_height = scaled_max_y - scaled_min_y
                    if region_width > 0 and region_height > 0:
                        detected_strong_anomaly_regions.append({
                            'bbox': (scaled_min_x, scaled_min_y, region_width, region_height),
                            'area_pixels': region.area
                        })
                    else:
                        print(f"    Región filtrada debido a dimensiones no válidas después del escalado: bbox({scaled_min_x}, {scaled_min_y}, {region_width}, {region_height})")

            print(f"    Se encontraron {len(detected_strong_anomaly_regions)} regiones anómalas 'fuertes' (área >= {min_region_pixel_area} píxeles) después de filtrar.")
            if len(detected_strong_anomaly_regions) > 0:
                print(f"    Ejemplo de región fuerte detectada (bbox - x_inicio, y_inicio, ancho, alto): {detected_strong_anomaly_regions[0]['bbox']}")
                plt.figure(figsize=(10, 8))
                plt.imshow(query_img_pil)
                plt.title(f'Imagen Anómala con Regiones Fuertes: {base_image_name.replace(".png", "")}')
                plt.axis('off')
                ax = plt.gca()
                for j, region_info in enumerate(detected_strong_anomaly_regions):
                    bbox = region_info['bbox']
                    if bbox[2] > 0 and bbox[3] > 0:
                        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                                 linewidth=3, edgecolor='lime', facecolor='none',
                                                 linestyle='-', alpha=0.9)
                        ax.add_patch(rect)
                    else:
                        print(f"    Saltando el dibujo de un bbox inválido: {bbox}")
                ax.add_patch(patches.Rectangle((0,0), 0.1, 0.1, linewidth=3, edgecolor='lime', facecolor='none', linestyle='-', alpha=0.9, label=f'Regiones Anómalas Fuertes'))
                plt.legend()
                strong_regions_overlay_output_filename = os.path.join(ANOMALY_REGIONS_SAVE_DIR, f'anomaly_regions_{base_image_name}')
                plt.tight_layout()
                plt.savefig(strong_regions_overlay_output_filename)
                print(f"    Plot de regiones anómalas fuertes guardado en: {strong_regions_overlay_output_filename}")
                plt.close()
            else:
                print("    No se generó el plot de regiones anómalas fuertes porque no se detectaron regiones válidas para dibujar.")
        end_time_region_detection = time.time()
        print(f"  Tiempo total para la detección y visualización de regiones para {base_image_name}: {end_time_region_detection - start_time_region_detection:.4f} segundos")
    else:
        print(f"  Clasificada como BUENA. No se dibujarán regiones anómalas para {base_image_name}.")

    directorio_imagenes = TRAIN_GOOD_DIR
    plot_save_directory_on_server = PLOT_SAVE_ROOT_DIR

    # --- Función para buscar imágenes similares usando KNN ---
    def buscar_imagenes_similares_knn(query_feature_map, pre_flattened_features_bank, k=3, nombres_archivos=None):
        query_feat_flatten = query_feature_map.flatten().cpu().numpy()

        features_bank_for_knn = pre_flattened_features_bank
        
        if isinstance(features_bank_for_knn, torch.Tensor):
            features_bank_for_knn = features_bank_for_knn.cpu().numpy()

        start_time_knn_dist = time.time()
        distances = euclidean_distances([query_feat_flatten], features_bank_for_knn)
        nearest_indices = np.argsort(distances[0])[:k]
        end_time_knn_dist = time.time()
        time_knn_dist = end_time_knn_dist - start_time_knn_dist
        print(f"Tiempo para calcular distancias KNN: {end_time_knn_dist - start_time_knn_dist:.4f} segundos")

        imagenes_similares = []
        rutas_imagenes_similares = []
        if nombres_archivos:
            for idx in nearest_indices:
                imagenes_similares.append(nombres_archivos[idx])
                rutas_imagenes_similares.append(os.path.join(directorio_imagenes, nombres_archivos[idx]))
        else:
            for idx in nearest_indices:
                imagenes_similares.append(f"Imagen_Banco_{idx:03d}.png")
                rutas_imagenes_similares.append(os.path.join(directorio_imagenes, f"Imagen_Banco_{idx:03d}.png"))

        print(f"Índices de los {k} vecinos más cercanos: {nearest_indices}")
        print("Imágenes similares:", imagenes_similares)
        return imagenes_similares, rutas_imagenes_similares, time_knn_dist

    # --- Uso de la función de búsqueda KNN con el banco de características pre-aplanado ---
    print("\nRealizando búsqueda KNN de imágenes similares usando el banco pre-aplanado del Coreset...")
    imagenes_similares, rutas_imagenes_similares, time_knn_dist = buscar_imagenes_similares_knn(
        query_lr_features, coreset_relevant_flat_features_bank, nombres_archivos=coreset_relevant_filenames
    )

    print("\n--- Resultados Finales de la Búsqueda KNN ---")
    print("Imágenes similares encontradas:", imagenes_similares)
    print("Rutas de imágenes similares:", rutas_imagenes_similares)

    # --- Visualización de las imágenes similares ---
    print("\nVisualizando las imágenes similares encontradas...")
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
        except FileNotFoundError:
            print(f"Advertencia: No se pudo encontrar la imagen en la ruta: {ruta_imagen_similar}")
            plt.subplot(1, len(rutas_imagenes_similares) + 1, j + 2)
            plt.text(0.5, 0.5, "Imagen no encontrada", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title(f'Vecino {j + 1}\n(No encontrada)')
            plt.axis('off')

    output_similar_plot_filename = os.path.join(plot_save_directory_on_server, f'similar_images_plot_{base_image_name}')
    plt.tight_layout()
    plt.savefig(output_similar_plot_filename)
    print(f"Plot de imágenes similares guardado en: {output_similar_plot_filename}")
    plt.close()
    print("Script Stage 2 completado para la imagen actual.")

    # --- 6. Aplicar FeatUp para obtener características de alta resolución ---

    def apply_featup_hr(image_path, featup_upsampler, image_transform, device):
        image_tensor = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            hr_feats = featup_upsampler(image_tensor)
            lr_feats = featup_upsampler.model(image_tensor)
        return lr_feats.cpu(), hr_feats.cpu()

    # 6.1 A imagen de consulta
    start_time_featup = time.time()
    input_query_tensor = transform(Image.open(query_image_path).convert("RGB")).unsqueeze(0).to(device)
    query_lr_feats_featup, query_hr_feats = apply_featup_hr(query_image_path, upsampler, transform, device)
    print(f"Dimensiones de características de referencia (baja resolución FeatUp): {query_lr_feats_featup.shape}")
    print(f"Dimensiones de características de referencia (alta resolución FeatUp): {query_hr_feats.shape}")
    end_time_featup = time.time()
    plot_feats(unnorm(input_query_tensor)[0], query_lr_feats_featup[0], query_hr_feats[0])
    output_query_plot_filename = os.path.join(plot_save_directory_on_server, f'query_image_features_plot_{base_image_name}')
    plt.tight_layout()
    plt.savefig(output_query_plot_filename)
    print(f"Plot de características de la imagen de consulta guardado en: {output_query_plot_filename}")
    plt.close()

    # 6.2 A similares imagenes
    similar_hr_feats_list = []
    print("Imágenes similares:", imagenes_similares)
    for j, similar_image_path in enumerate(rutas_imagenes_similares):
        input_similar_tensor = transform(Image.open(similar_image_path).convert("RGB")).unsqueeze(0).to(device)
        
        similar_lr_feats, similar_hr_feats = apply_featup_hr(similar_image_path, upsampler, transform, device)
        similar_hr_feats_list.append(similar_hr_feats)
        plt.figure(figsize=(10, 5))
        plot_feats(unnorm(input_similar_tensor)[0], similar_lr_feats[0], similar_hr_feats[0])
        
        output_similar_plot_filename = os.path.join(plot_save_directory_on_server, f'similar_image_{j + 1}_features_plot_{base_image_name}')
        plt.tight_layout()
        plt.savefig(output_similar_plot_filename)
        print(f"Plot de características de la imagen similar guardado en: {output_similar_plot_filename}")
        plt.close()

    ########################
    # APLICANDO SAM MASK (ONLY IF CLASSIFIED AS ANOMALOUS AND SAM MODEL IS LOADED)
    if classification == 1 and sam2_model is not None:
        print(f"\nIniciando SAM para la imagen anómala: {base_image_name}")
        start_time_sam = time.time()

        def show_mask(mask, ax, random_color=False, borders = True):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            
            mask_image_alpha = np.zeros((h, w, 4), dtype=np.float32)
            mask_image_alpha[mask > 0] = color

            if borders:
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                contour_image = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=2)
                
                contour_image_float = contour_image.astype(np.float32) / 255.0
                contour_mask = (contour_image_float.sum(axis=-1) > 0).astype(np.float32)
                
                mask_image_alpha[contour_mask > 0, :3] = 1.0
                mask_image_alpha[contour_mask > 0, 3] = 0.5

            ax.imshow(mask_image_alpha)

        def show_points(coords, labels, ax, marker_size=375):
            pos_points = coords[labels==1]
            neg_points = coords[labels==0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

        def show_masks_grid(image, masks, points=None, plot_title="Generated Masks", ax=None, num_masks=0):
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image)
            
            if points is not None:
                point_labels_for_display = np.ones(points.shape[0], dtype=int)
                show_points(points, point_labels_for_display, ax, marker_size=50)

            for mask_data in masks:
                mask = mask_data["segmentation"]
                show_mask(mask, ax, random_color=True)
            
            ax.set_title(f"{plot_title} (Masks: {num_masks})", fontsize=18)
            ax.axis('off')

        try:
            image_for_sam = Image.open(query_image_path).convert("RGB")
            image_for_sam_np = np.array(image_for_sam)
            print(f"Dimensiones imagen de entrada a SAM (np.array(query_img_pil)): {image_for_sam_np.shape}")
        except Exception as e:
            print(f"Error al procesar la imagen de consulta para SAM: {e}")
            print(f"Saltando SAM para la imagen {base_image_name} debido a error al abrir la imagen.")
            continue

        # --- Parámetros para el grid de puntos y filtrado de máscaras ---
        points_grid_density = 16
        min_mask_area_pixels = 100.0
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

        print(f"Generando máscaras con un grid de {points_grid_density}x{points_grid_density} puntos...")

        masks_data_query_image = mask_generator_query.generate(image_for_sam_np)
        masks_data_query_image = [mask_info for mask_info in masks_data_query_image if mask_info['area'] <= max_mask_area_pixels]
        print(f"Se generaron {len(masks_data_query_image)} máscaras después de filtrar por área máxima para {base_image_name}.")
        
        all_generated_points = []
        for mask_info in masks_data_query_image:
            if "point_coords" in mask_info:
                all_generated_points.append(mask_info["point_coords"])
        if all_generated_points:
            all_generated_points = np.concatenate(all_generated_points, axis=0)
        else:
            all_generated_points = None

        show_masks_grid(image_for_sam_np, masks_data_query_image, points=all_generated_points, 
                        plot_title=f"SAM2 Masks for Query: {base_image_name.replace('.png', '')}",
                        num_masks=len(masks_data_query_image))
        output_query_grid_mask_filename = os.path.join(plot_save_directory_on_server, f'sam_query_image_grid_masks_{base_image_name}')
        plt.savefig(output_query_grid_mask_filename)
        print(f"Plot de máscaras del grid para la imagen de consulta guardado en: {output_query_grid_mask_filename}")
        plt.close()

        mask_generator_similar = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=points_grid_density,
            points_per_batch=256,
            pred_iou_thresh=0.48,
            stability_score_thresh=0.7,
            crop_n_layers=0,
            min_mask_region_area=800,
        )

        print("\nGenerando y visualizando máscaras SAM para las imágenes similares...")
        similar_masks_raw_list = []
        for j, similar_image_path in enumerate(rutas_imagenes_similares):
            try:
                img_similar_pil = Image.open(similar_image_path).convert('RGB')
                image_np_similar_for_sam = np.array(img_similar_pil)

                print(f"--- Procesando vecino {j+1}: {os.path.basename(similar_image_path)} ---")
                current_similar_masks_data = mask_generator_similar.generate(image_np_similar_for_sam)
                
                current_similar_masks_data = [
                    mask_info for mask_info in current_similar_masks_data 
                    if mask_info['area'] <= max_mask_area_pixels
                ]
                
                similar_masks_raw_list.append(current_similar_masks_data)
                print(f"Dimensiones imagen similar para SAM (np.array): {image_np_similar_for_sam.shape}")

                all_generated_points_similar = []
                for mask_info in current_similar_masks_data:
                    if "point_coords" in mask_info:
                        all_generated_points_similar.append(mask_info["point_coords"])
                if all_generated_points_similar:
                    all_generated_points_similar = np.concatenate(all_generated_points_similar, axis=0)
                else:
                    all_generated_points_similar = None

                show_masks_grid(image_np_similar_for_sam, current_similar_masks_data, 
                                points=all_generated_points_similar, 
                                plot_title=f"SAM2 Masks - Vecino {j + 1} ({os.path.basename(similar_image_path)})",
                                num_masks=len(current_similar_masks_data))
                
                output_similar_grid_mask_filename = os.path.join(plot_save_directory_on_server, f'sam_similar_image_{j + 1}_grid_masks_{base_image_name}')
                plt.savefig(output_similar_grid_mask_filename)
                print(f"Plot de máscaras del grid para el vecino {j + 1} guardado en: {output_similar_grid_mask_filename}")
                plt.close()

            except Exception as e:
                print(f"Error al procesar la imagen similar {os.path.basename(similar_image_path)} para la generación de máscaras SAM: {e}")

        end_time_sam = time.time()
        print(f"Tiempo de ejecución de SAM con grid: {end_time_sam - start_time_sam:.4f} segundos para {base_image_name}")

        print("\nGenerando visualización combinada de imágenes originales y segmentadas...")

        combined_plots_directory = os.path.join(plot_save_directory_on_server, "combined_segmented_plots")
        os.makedirs(combined_plots_directory, exist_ok=True)
        print(f"Carpeta para plots combinados creada en: {combined_plots_directory}")

        def plot_combined_segmented(query_original_path, query_masks, similar_original_paths, similar_masks_list, output_dir, current_image_name):
            num_similar = len(similar_original_paths)
            if num_similar == 0:
                print("No se encontraron imágenes similares para la visualización combinada.")
                return

            total_subplots = 2 + num_similar
            fig, axes = plt.subplots(1, total_subplots, figsize=(5 * total_subplots, 6))

            query_img_orig = None
            try:
                query_img_orig = Image.open(query_original_path).convert('RGB')
                axes[0].imshow(query_img_orig)
                axes[0].set_title(f'Consulta Original:\n{current_image_name.replace(".png", "")}')
                axes[0].axis('off')
            except Exception as e:
                print(f"Error al cargar la imagen de consulta original: {e}")
                axes[0].text(0.5, 0.5, "Error de Carga", horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
                axes[0].set_title('Consulta Original (Error)')
                axes[0].axis('off')

            if query_img_orig is not None:
                image_for_sam_np_query = np.array(query_img_orig)
                show_masks_grid(image_for_sam_np_query, query_masks, plot_title=f'Consulta Segmentada', ax=axes[1], num_masks=len(query_masks))
            else:
                axes[1].text(0.5, 0.5, "Consulta Segmentada (No Imagen)", horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
                axes[1].set_title('Consulta Segmentada')
                axes[1].axis('off')

            for j, similar_path in enumerate(similar_original_paths):
                if j + 2 >= total_subplots:
                    break 

                try:
                    similar_img_orig = Image.open(similar_path).convert('RGB')
                    image_np_similar_for_sam = np.array(similar_img_orig)
                    
                    current_similar_masks = []
                    if j < len(similar_masks_list):
                        current_similar_masks = similar_masks_list[j]
                    
                    show_masks_grid(image_np_similar_for_sam, current_similar_masks, 
                                    plot_title=f'Vecino {j+1} Segmentado\n({os.path.basename(similar_path)})', 
                                    ax=axes[j + 2],
                                    num_masks=len(current_similar_masks))
                    
                except Exception as e:
                    print(f"Error al graficar la imagen similar {os.path.basename(similar_path)}: {e}")
                    axes[j + 2].text(0.5, 0.5, "Error al Graficar", horizontalalignment='center', verticalalignment='center', transform=axes[j + 2].transAxes)
                    axes[j + 2].set_title(f'Vecino {j+1} (Error)')
                    axes[j + 2].axis('off')

            plt.tight_layout()
            output_filename = os.path.join(output_dir, f'combined_query_and_similar_segmented_{current_image_name}')
            plt.savefig(output_filename)
            print(f"Plot combinado de imágenes segmentadas guardado en: {output_filename}")
            plt.close(fig)

        plot_combined_segmented(
            query_image_path, 
            masks_data_query_image,
            rutas_imagenes_similares,
            similar_masks_raw_list,
            combined_plots_directory,
            base_image_name
        )
    else:
        print(f"La imagen {base_image_name} fue clasificada como BUENA o el modelo SAM no se pudo cargar. No se generarán máscaras SAM ni plots combinados de segmentación.")


print("\nAnálisis de detección de anomalías por lotes y métricas completado.")

# --- Salida final del vector de clasificación ---
print("\n--- Clasificación Final de Anomalías por Imagen ---")
for i, (img_name, classification_result) in enumerate(zip(image_names_processed, image_classifications)):
    status = "Anómala" if classification_result == 1 else "Buena"
    print(f"   {i+1}. Imagen: {img_name} -> Clasificación: {status} ({classification_result})")

print("\nVector de clasificaciones (0=Buena, 1=Anómala):")
print(image_classifications)