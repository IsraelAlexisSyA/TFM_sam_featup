import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
# from sklearn.metrics.pairwise import euclidean_distances # Not directly used in final logic for distances
import time

# Import FeatUp utilities for normalization/denormalization
from featup.util import norm, unnorm
# from featup.plotting import plot_feats # Not used in this specific plotting task

from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
# from scipy.cluster.hierarchy import dendrogram, linkage # Not requested for batch plotting
# from scipy.spatial.distance import pdist # Not requested for batch plotting
# from sklearn.cluster import KMeans # Not requested for batch plotting

from scipy.stats import median_abs_deviation

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224 # Input size for DINOv2
BACKBONE_PATCH_SIZE = 14 # Patch size for DINOv2 ViT-S/14
use_norm = True # Consistent with your approach

# Spatial dimensions of low-resolution feature maps (H', W')
H_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 14 = 16
W_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 14 = 16

# Directories
TRAIN_GOOD_DIR = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/train/good'
TEST_CRACK_DIR = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/test/good' # Your target directory
PLOT_SAVE_ROOT_DIR = '/home/imercatoma/FeatUp/plots_anomaly_distances' # New root for all distance plots
os.makedirs(PLOT_SAVE_ROOT_DIR, exist_ok=True)
print(f"Root plot save directory created/verified: '{PLOT_SAVE_ROOT_DIR}'")

# Paths for Coreset files (from Stage 1/previous steps)
core_bank_filenames_file = os.path.join(TRAIN_GOOD_DIR, 'core_bank_filenames.pt')
coreset_relevant_flat_features_bank_file = os.path.join(TRAIN_GOOD_DIR, 'coreset_relevant_flat_features_bank.pt')
template_features_bank_coreset_file = os.path.join(TRAIN_GOOD_DIR, 'template_features_bank_coreset.pt') # This is your M

# --- Load Coreset Data (The 'M' matrix for KNN) ---
print("Loading relevant coreset data and feature bank (M)...")
coreset_relevant_filenames = []
coreset_relevant_flat_features_bank = None
coreset_features = None # This will be the actual 'M' matrix for KNN

try:
    coreset_relevant_filenames = torch.load(core_bank_filenames_file)
    coreset_relevant_flat_features_bank = torch.load(coreset_relevant_flat_features_bank_file).to(device)
    coreset_features = torch.load(template_features_bank_coreset_file).to(device) # Your 'M'
    
    print(f"Coreset of features (M) loaded. Dimension: {coreset_features.shape}")
    print(f"Coreset relevant flat features bank loaded. Dimension: {coreset_relevant_flat_features_bank.shape}")
    print(f"Number of relevant filenames loaded: {len(coreset_relevant_filenames)}")

except FileNotFoundError as e:
    print(f"Error loading coreset files: {e}. Ensure Stage 1 was executed and files exist.")
    exit()
except Exception as e:
    print(f"An error occurred loading or processing coreset files: {e}")
    exit()

# Move coreset to CPU for sklearn's NearestNeighbors
coreset_features_cpu = coreset_features.cpu().numpy()
print(f"Coreset features moved to CPU. Shape: {coreset_features_cpu.shape}")

# Initialize NearestNeighbors finder once
nn_finder = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(coreset_features_cpu)
print("NearestNeighbors finder initialized with coreset features.")

# --- Load DINOv2 Model ---
print("Loading DINOv2 model for feature extraction...")
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
dinov2_model = upsampler.model # Get the base DINOv2 model from the upsampler
dinov2_model.eval() # Set model to evaluation mode
print("DINOv2 model loaded.")

# --- Image Transformation ---
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(), # Scales pixels to [0, 1] and changes to (C, H, W)
    norm # Applies ImageNet normalization (mean/std)
])

# --- Core Function to Get Anomaly Scores for an Image ---
def get_anomaly_scores_for_image(image_path, model, image_transform, nn_finder_instance, H_prime, W_prime, device):
    """
    Extracts DINOv2 features for a single image, computes patch anomaly scores
    based on distance to coreset.
    Returns sorted_patch_anomaly_scores.
    """
    # 1. Extract DINOv2 features for the query image
    try:
        input_tensor = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"  Error loading/transforming image {os.path.basename(image_path)}: {e}")
        return None

    with torch.no_grad():
        features_lr = model(input_tensor) # (1, C, H', W')

    # 2. Flatten patches
    query_patches_flat = features_lr.squeeze(0).permute(1, 2, 0).reshape(-1, features_lr.shape[1])
    query_patches_flat_cpu = query_patches_flat.cpu().numpy()

    # 3. Calculate distances to nearest neighbors in coreset
    distances_to_nn, _ = nn_finder_instance.kneighbors(query_patches_flat_cpu)
    patch_anomaly_scores = distances_to_nn.flatten() # (H'*W',)

    # 4. Sort anomaly scores in descending order
    sorted_patch_anomaly_scores = np.sort(patch_anomaly_scores)[::-1]

    # Ensure it's 256 elements for consistency, though not strictly enforced in your previous code
    # if sorted_patch_anomaly_scores.shape[0] != 256:
    #     print(f"  Warning: Anomaly scores count is {sorted_patch_anomaly_scores.shape[0]}, not 256.")

    return sorted_patch_anomaly_scores

# --- Metric Calculation Functions (from previous steps) ---
def calculate_rms(data):
    """Calculates the Root Mean Square (RMS) of an array of data."""
    return np.sqrt(np.mean(data**2))

def calculate_mad(data):
    """Calculates the Median Absolute Deviation (MAD) of an array of data."""
    return median_abs_deviation(data)

def calculate_median(data):
    """Calculates the Median of an array of data."""
    return np.median(data)

def calculate_quartile(data, q=25):
    """Calculates a specific quartile (e.g., 25th percentile for Q1)."""
    return np.percentile(data, q)


# --- BATCH PROCESSING AND DATA COLLECTION ---
image_names_processed = []
rms_mad_distances = []
rms_median_distances = []
rms_q1_distances = []

print(f"\nStarting batch processing of images from: {TEST_CRACK_DIR}")
try:
    test_image_files = [f for f in os.listdir(TEST_CRACK_DIR) if f.lower().endswith('.png')]
    test_image_files.sort() # Ensure consistent order
    if not test_image_files:
        print(f"No .png images found in: {TEST_CRACK_DIR}")
except FileNotFoundError:
    print(f"Error: Directory '{TEST_CRACK_DIR}' not found.")
    test_image_files = []

for img_file in test_image_files:
    full_image_path = os.path.join(TEST_CRACK_DIR, img_file)
    print(f"Processing image: {img_file}")

    # Get sorted anomaly scores for the current image
    current_sorted_patch_anomaly_scores = get_anomaly_scores_for_image(
        full_image_path, dinov2_model, transform, nn_finder, H_prime, W_prime, device
    )

    if current_sorted_patch_anomaly_scores is None:
        print(f"  Skipping {img_file} due to processing error.")
        continue # Skip to next image if error occurred

    # --- Normalize Anomaly Scores ---
    min_val = np.min(current_sorted_patch_anomaly_scores)
    max_val = np.max(current_sorted_patch_anomaly_scores)

    if max_val == min_val:
        normalized_data = np.zeros_like(current_sorted_patch_anomaly_scores, dtype=float)
        # print("  Warning: All anomaly scores are identical for this image. Normalized data will be all zeros.")
    else:
        normalized_data = (current_sorted_patch_anomaly_scores - min_val) / (max_val - min_val)

    # --- Calculate Metrics from Normalized Data ---
    A_rms = calculate_rms(normalized_data)
    B_mad = calculate_mad(normalized_data)
    C_median = calculate_median(normalized_data)
    D_q1 = calculate_quartile(normalized_data, q=25) # 1st Quartile

    # --- Calculate Distances ---
    dist_rms_mad = A_rms - B_mad
    dist_rms_median = A_rms - C_median
    dist_rms_q1 = A_rms - D_q1

    # Store results
    image_names_processed.append(img_file)
    rms_mad_distances.append(dist_rms_mad)
    rms_median_distances.append(dist_rms_median)
    rms_q1_distances.append(dist_rms_q1)

    print(f"  Distances calculated for {img_file}:")
    print(f"    (RMS - MAD): {dist_rms_mad:.4f}")
    print(f"    (RMS - Mediana): {dist_rms_median:.4f}")
    print(f"    (RMS - 1er Cuartil): {dist_rms_q1:.4f}")


# --- Plotting Results ---

if not image_names_processed:
    print("\nNo image data was successfully processed for plotting.")
else:
    # Determine figure width dynamically based on number of images
    num_images = len(image_names_processed)
    fig_width = max(12, num_images * 0.8) # Ensure reasonable minimum width, scale for more images
    x_positions = np.arange(num_images)

    # --- Plot 1: RMS - MAD Distances ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, rms_mad_distances, color='purple', width=0.6)
    plt.ylabel('Distance (RMS - MAD)', fontsize=12)
    plt.title('RMS - MAD Distance per Image (Test: hazelnut/crack)', fontsize=14)
    plt.xticks(x_positions, image_names_processed, rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    output_path_rms_mad = os.path.join(PLOT_SAVE_ROOT_DIR, 'distances_rms_mad_per_test_image.png')
    plt.savefig(output_path_rms_mad)
    print(f"\nPlot 'RMS - MAD per Image' saved to: '{output_path_rms_mad}'")
    plt.close()

    # --- Plot 2: RMS - Mediana Distances ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, rms_median_distances, color='orange', width=0.6)
    plt.ylabel('Distance (RMS - Mediana)', fontsize=12)
    plt.title('RMS - Mediana Distance per Image (Test: hazelnut/crack)', fontsize=14)
    plt.xticks(x_positions, image_names_processed, rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    output_path_rms_median = os.path.join(PLOT_SAVE_ROOT_DIR, 'distances_rms_median_per_test_image.png')
    plt.savefig(output_path_rms_median)
    print(f"Plot 'RMS - Mediana per Image' saved to: '{output_path_rms_median}'")
    plt.close()

    # --- Plot 3: RMS - 1st Quartile (Q1) Distances ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, rms_q1_distances, color='cyan', width=0.6)
    plt.ylabel('Distance (RMS - 1er Cuartil)', fontsize=12)
    plt.title('RMS - 1er Cuartil Distance per Image (Test: hazelnut/crack)', fontsize=14)
    plt.xticks(x_positions, image_names_processed, rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    output_path_rms_q1 = os.path.join(PLOT_SAVE_ROOT_DIR, 'distances_rms_q1_per_test_image.png')
    plt.savefig(output_path_rms_q1)
    print(f"Plot 'RMS - 1er Cuartil per Image' saved to: '{output_path_rms_q1}'")
    plt.close()

print("\nBatch anomaly detection and metric analysis complete.")





exit()









################# saber si existe o no anomalias con tecnicas DWT y codo en pendiente #################
import numpy as np
import pywt # Importar la librería PyWavelets
import matplotlib.pyplot as plt
import os
from kneed import KneeLocator # Importar KneeLocator
# Simulación de plot_save_directory_on_server
# Reemplaza con tu ruta real donde quieras guardar las imágenes
#plot_save_directory_on_server = 'plots_wavelet'




print("Top scores de anomalía (1% superior):")
print(top_anomaly_scores)
# Ordenar los patch_anomaly_scores de forma descendente
sorted_patch_anomaly_scores = np.sort(patch_anomaly_scores)[::-1]

print("Top scores de anomalía (primeros 12, para referencia):")
print(sorted_patch_anomaly_scores[:12])
print(f"Total de scores: {len(sorted_patch_anomaly_scores)}")

# --- Preservar el score máximo original antes de normalizar ---
original_max_score = sorted_patch_anomaly_scores[0]
print(f"\nScore máximo original (sin normalizar): {original_max_score:.6f}")


# --- ESTRATEGIA A: NORMALIZACIÓN DE LOS SCORES ---
# Normalizar los scores al rango [0, 1]
if sorted_patch_anomaly_scores.max() > sorted_patch_anomaly_scores.min():
    normalized_scores = (sorted_patch_anomaly_scores - sorted_patch_anomaly_scores.min()) / \
                        (sorted_patch_anomaly_scores.max() - sorted_patch_anomaly_scores.min())
    print("Scores de anomalía normalizados al rango [0, 1].")
else:
    normalized_scores = np.zeros_like(sorted_patch_anomaly_scores)
    print("Todos los scores de anomalía son iguales, normalizados a cero.")

# A partir de aquí, usa normalized_scores para todos los cálculos
current_scores_for_analysis = normalized_scores

# --- Aplicar la Transformada Wavelet Discreta (DWT) ---
wavelet = 'db4'
level = 1
coeffs = pywt.wavedec(current_scores_for_analysis, wavelet, level=level)

if level == 1:
    cD1 = coeffs[1]
else:
    cD1 = coeffs[-1]

print(f"\nNúmero de coeficientes de detalle (cD1): {len(cD1)}")
print(f"Los 10 primeros coeficientes de detalle (cD1): {cD1[:10]}")

# --- Calcular métricas de los coeficientes de detalle ---
N_coeffs_to_consider = min(len(cD1), 20) # Considerar los primeros 20 coefs o menos si cD1 es más corto
abs_sum_cD1 = np.sum(np.abs(cD1[:N_coeffs_to_consider]))
energy_cD1 = np.sum(cD1[:N_coeffs_to_consider]**2)
max_abs_cD1 = np.max(np.abs(cD1[:N_coeffs_to_consider]))

print(f"\nCaracterísticas de cD1 (primeros {N_coeffs_to_consider} coeficientes) de scores normalizados:")
print(f"   Suma Absoluta (abs_sum_cD1): {abs_sum_cD1:.6f}")
print(f"   Energía (energy_cD1): {energy_cD1:.6f}")
print(f"   Máximo Absoluto (max_abs_cD1): {max_abs_cD1:.6f}")

# --- Detección de "Codo" basada en cambio de pendiente ---
elbow_point_index = None
length_steep_slope = 0

search_range = min(len(current_scores_for_analysis), 100) # Buscar el codo en los primeros 100 puntos
scores_to_analyze_for_slope = current_scores_for_analysis[:search_range]
x_indices = np.arange(search_range)

if len(scores_to_analyze_for_slope) > 1:
    slopes = np.abs(np.diff(scores_to_analyze_for_slope))

    if len(slopes) > 0:
        initial_slope_points = min(5, len(slopes))
        max_initial_slope = np.mean(slopes[:initial_slope_points])
        
        # --- AJUSTE CLAVE AQUÍ: Definir slope_threshold_multiplier condicionalmente ---
        slope_threshold_multiplier = 0.2 # Valor predeterminado
        
        # Si la primera pendiente es extremadamente pequeña en relación con el promedio inicial,
        # lo que sugiere una estabilización casi inmediata (length_steep_slope sería 1).

        # Si la primera pendiente es muy pequeña en relación con el promedio inicial,
        # lo que sugiere una estabilización casi inmediata (length_steep_slope sería 1).
        if max_initial_slope > 1e-6: # Evitar división por cero
            # Aumentamos el umbral para que la condición sea TRUE más fácilmente
            # Por ejemplo, si slopes[0] es menor que el 10% del promedio inicial
            if slopes[0] < (max_initial_slope * 0.10): # Ajustado de 0.01 a 0.10
                slope_threshold_multiplier = 0.02 # O 0.001 si esa es tu intención final
                print(f"  --> Ajustando slope_threshold_multiplier a {slope_threshold_multiplier:.2f} (caída inicial muy abrupta).")
 
        
        slope_threshold = max_initial_slope * slope_threshold_multiplier
        
        print(f"\nCalculando codo por umbral de pendiente (en scores normalizados):")
        print(f"   Pendiente inicial promedio ({initial_slope_points} pts): {max_initial_slope:.6f}")
        print(f"   Umbral de pendiente para detección de codo: {slope_threshold:.6f}")
        print(f"   Multiplicador de umbral de pendiente usado: {slope_threshold_multiplier:.2f}")

        print(f"\nPrimeros 5 scores normalizados: {current_scores_for_analysis[:5]}")
        if len(slopes) > 0:
            print(f"Primeros 5 valores de pendientes (abs): {slopes[:5]}")
            print(f"max_initial_slope: {max_initial_slope:.6f}")
            print(f"slope_threshold actual: {slope_threshold:.6f}")
        

        for i in range(len(slopes)):
            if slopes[i] < slope_threshold:
                elbow_point_index = i + 1
                break
    
    if elbow_point_index is not None:
        length_steep_slope = elbow_point_index
        print(f"Codo detectado (Umbral de Pendiente): {elbow_point_index}, Longitud: {length_steep_slope}")
    else:
        length_steep_slope = len(scores_to_analyze_for_slope) 
        print(f"No se detectó un punto de 'codo' claro con el umbral de pendiente.")
        print(f"Longitud estimada de la pendiente inclinada (asumiendo toda la porción analizada): {length_steep_slope} puntos")
else:
    length_steep_slope = 0
    print("\nNo hay suficientes scores para calcular la pendiente para la detección de codo.")

if elbow_point_index is None and len(scores_to_analyze_for_slope) <= 1:
    elbow_point_index = None
    length_steep_slope = 0

# --- Calcular el score original en la ubicación del codo ---
score_at_elbow_original = None
if elbow_point_index is not None and elbow_point_index < len(sorted_patch_anomaly_scores):
    score_at_elbow_original = sorted_patch_anomaly_scores[elbow_point_index]
    print(f"Score en el punto del codo (Original): {score_at_elbow_original:.6f}")
else:
    score_at_elbow_original = float('inf') 

# --- Visualizar los Coeficientes de Detalle (cD1) y el Codo ---
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(current_scores_for_analysis, label='Scores de Anomalía Ordenados (Normalizados)', color='blue')
if elbow_point_index is not None and elbow_point_index < len(current_scores_for_analysis):
    plt.axvline(x=elbow_point_index, color='green', linestyle='--', label=f'Codo detectado (índice: {elbow_point_index})')
    plt.plot(elbow_point_index, current_scores_for_analysis[elbow_point_index], 'go', markersize=8)
plt.title("Scores de Anomalía Ordenados (Normalizados) con Codo Detectado")
plt.xlabel("Índice")
plt.ylabel("Score Normalizado")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
x_axis_cD1 = np.linspace(0, len(current_scores_for_analysis) - 1, len(cD1))
plt.plot(x_axis_cD1, cD1, label=f'Coeficientes de Detalle (cD1, Wavelet: {wavelet})', color='red')
plt.axhline(y=0.01, color='purple', linestyle=':', label='Mini-umbral 0.01 (referencia para cD1)')
plt.axhline(y=-0.01, color='purple', linestyle=':') 
plt.title(f"Coeficientes de Detalle (cD1) de la Transformada Wavelet Discreta")
plt.xlabel("Posición aproximada en la secuencia original")
plt.ylabel("Amplitud del Coeficiente de Detalle")
plt.grid(True)
plt.legend()

plt.tight_layout()

output_dwt_coeffs_plot_filename = os.path.join(plot_save_directory_on_server, 'dwt_detail_coefficients_normalized_with_elbow.png')
plt.savefig(output_dwt_coeffs_plot_filename)
print(f"\nPlot de los coeficientes de detalle DWT y codo guardado en: {output_dwt_coeffs_plot_filename}")
plt.close()

# --- Consolidar resultados para clasificación ---
analysis_results = {
    'original_max_score': original_max_score, 
    'max_score': current_scores_for_analysis[0],
    'abs_sum_cD1': abs_sum_cD1,
    'energy_cD1': energy_cD1,
    'max_abs_cD1': max_abs_cD1,
    'elbow_point_index': elbow_point_index,
    'length_steep_slope': length_steep_slope,
    'score_at_elbow_original': score_at_elbow_original,
}

print("\n--- Resultados del Análisis (incluye Original Max Score) ---")
for key, value in analysis_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")


# --- Lógica de Clasificación con la nueva estrategia ---

def classify_anomaly_type(results: dict):
    original_max_score = results['original_max_score']
    abs_sum_cD1 = results['abs_sum_cD1']
    length_steep_slope = results['length_steep_slope']
    max_abs_cD1 = results['max_abs_cD1']
    score_at_elbow_original = results['score_at_elbow_original']

    # --- UMBRALES CLAVE DE CLASIFICACIÓN (AJUSTAR CUIDADOSAMENTE) ---
    THRESHOLD_ORIGINAL_MAX_LARGE = 0.30
    THRESHOLD_ORIGINAL_MAX_MILD_MIN = 0.15 
    
    THRESHOLD_ORIGINAL_MAX_VERY_MILD_GOOD_THRESHOLD = 0.17 

    THRESHOLD_LENGTH_STEEP_GOOD_VS_ANOMALY = 10 
    
    # Este umbral no se usa directamente en la lógica de clasificación con el cambio actual,
    # ya que la decisión de "caída extremadamente rápida" ahora se basa más en el THRESHOLD_SCORE_AT_ELBOW_GOOD
    # y la lógica del codo.
    # THRESHOLD_EXTREMELY_RAPID_FALL_GOOD = 2 

    # Los demás umbrales se mantienen como los indicados en tu último log o por defecto
    THRESHOLD_CONCENTRATION_FOR_SHAPE_DESCRIPTOR = 0.25
    THRESHOLD_CONCENTRATION_HIGH_FOR_GOOD_INDICATOR = 0.60 
    THRESHOLD_CD1_ACTIVITY_VERY_LOW_GOOD = 0.08 
    
    # Ajustado de 0.30 a 0.20
    THRESHOLD_SCORE_AT_ELBOW_GOOD = 0.20 
    
    # Inicializar is_elbow_score_below_threshold antes de usarla
    is_elbow_score_below_threshold = False 
    if score_at_elbow_original is not None and score_at_elbow_original != float('inf'):
        is_elbow_score_below_threshold = (score_at_elbow_original < THRESHOLD_SCORE_AT_ELBOW_GOOD)


    # Calcular la nueva métrica de "concentración de cambio"
    concentration_of_change = max_abs_cD1 / abs_sum_cD1 if abs_sum_cD1 != 0 else 0

    print(f"\n--- Clasificando ---")
    print(f"  Original Max Score: {original_max_score:.6f}")
    print(f"  Suma Abs cD1 (Normalizado): {abs_sum_cD1:.6f}")
    print(f"  Máx Abs cD1 (Normalizado): {max_abs_cD1:.6f}")
    print(f"  Concentración de Cambio (MaxAbsCD1/AbsSumCD1): {concentration_of_change:.6f}")
    print(f"  Longitud Pendiente (Ubicación del Codo): {length_steep_slope}")
    print(f"  Score en el Codo (Original): {score_at_elbow_original:.6f}") 

    # --- Generar el descriptor de forma (para integrar directamente en la frase) ---
    is_concentrated_shape = concentration_of_change > THRESHOLD_CONCENTRATION_FOR_SHAPE_DESCRIPTOR
    shape_type_adjective_profile = "concentrado" if is_concentrated_shape else "distribuido"


    # --- LÓGICA DE CLASIFICACIÓN ---
    print("\n--- Evaluación de Condiciones ---")
    print(f"original_max_score ({original_max_score:.6f}) > THRESHOLD_ORIGINAL_MAX_LARGE ({THRESHOLD_ORIGINAL_MAX_LARGE:.2f}): {original_max_score > THRESHOLD_ORIGINAL_MAX_LARGE}")
    print(f"original_max_score ({original_max_score:.6f}) < THRESHOLD_ORIGINAL_MAX_MILD_MIN ({THRESHOLD_ORIGINAL_MAX_MILD_MIN:.2f}): {original_max_score < THRESHOLD_ORIGINAL_MAX_MILD_MIN}")
    print(f"length_steep_slope ({length_steep_slope}) <= THRESHOLD_LENGTH_STEEP_GOOD_VS_ANOMALY ({THRESHOLD_LENGTH_STEEP_GOOD_VS_ANOMALY}): {length_steep_slope <= THRESHOLD_LENGTH_STEEP_GOOD_VS_ANOMALY}")
    print(f"is_elbow_score_below_threshold ({score_at_elbow_original:.6f} < {THRESHOLD_SCORE_AT_ELBOW_GOOD:.2f}): {is_elbow_score_below_threshold}")
    print(f"abs_sum_cD1 ({abs_sum_cD1:.6f}) < THRESHOLD_CD1_ACTIVITY_VERY_LOW_GOOD ({THRESHOLD_CD1_ACTIVITY_VERY_LOW_GOOD:.2f}): {abs_sum_cD1 < THRESHOLD_CD1_ACTIVITY_VERY_LOW_GOOD}")
    print(f"concentration_of_change ({concentration_of_change:.6f}) > THRESHOLD_CONCENTRATION_HIGH_FOR_GOOD_INDICATOR ({THRESHOLD_CONCENTRATION_HIGH_FOR_GOOD_INDICATOR:.2f}): {concentration_of_change > THRESHOLD_CONCENTRATION_HIGH_FOR_GOOD_INDICATOR}")
    print(f"concentration_of_change ({concentration_of_change:.6f}) > THRESHOLD_CONCENTRATION_FOR_SHAPE_DESCRIPTOR ({THRESHOLD_CONCENTRATION_FOR_SHAPE_DESCRIPTOR:.2f}): {concentration_of_change > THRESHOLD_CONCENTRATION_FOR_SHAPE_DESCRIPTOR}")


    # 1. Anomalía GRANDE (Score de pico original muy alto)
    if original_max_score > THRESHOLD_ORIGINAL_MAX_LARGE:
        print("Entra en: Anomalía Grande (Pico muy alto)")
        return f"Anomalía Grande (Pico muy alto, perfil {shape_type_adjective_profile})"

    # 2. IMAGEN BUENA (Score de pico original BAJO - ¡Se da por sentado!)
    elif original_max_score < THRESHOLD_ORIGINAL_MAX_MILD_MIN: 
        print("  --> Entra en la condición de 'Buena' por score original BAJO (incondicional).")
        return f"Buena (Score pico muy bajo, perfil {shape_type_adjective_profile} y baja actividad general)"
            
    # 3. RANGO INTERMEDIO (0.15 a 0.30) - Aquí se subdivide
    elif original_max_score >= THRESHOLD_ORIGINAL_MAX_MILD_MIN and original_max_score <= THRESHOLD_ORIGINAL_MAX_LARGE:
        print("  --> Dentro del rango de scores originales intermedios (0.15-0.30).")

        # Sub-clasificación para scores MUY BAJOS dentro del rango intermedio (0.15 a < 0.17)
        if original_max_score < THRESHOLD_ORIGINAL_MAX_VERY_MILD_GOOD_THRESHOLD: 
            print(f"  --> Dentro del sub-rango intermedio muy bajo (< {THRESHOLD_ORIGINAL_MAX_VERY_MILD_GOOD_THRESHOLD:.2f}).")
            if is_elbow_score_below_threshold:
                return f"Buena (Pico muy bajo en rango intermedio, perfil {shape_type_adjective_profile} - codo temprano y score bajo)"
            else:
                return f"Límite: Falla muy pequeña (Pico muy bajo, perfil {shape_type_adjective_profile} - codo no tan bajo)"
        
        # Clasificación para el resto del rango intermedio (0.17 a 0.30)
        else: # original_max_score >= THRESHOLD_ORIGINAL_MAX_VERY_MILD_GOOD_THRESHOLD
            print(f"  --> Dentro del sub-rango intermedio medio/alto (>= {THRESHOLD_ORIGINAL_MAX_VERY_MILD_GOOD_THRESHOLD:.2f}).")
            
            # La condición para "caída extremadamente rápida" ahora se gestiona por la precisión del codo y el umbral de score.
            # Se ha eliminado el 'THRESHOLD_EXTREMELY_RAPID_FALL_GOOD' como una prioridad separada aquí
            # para evitar contradicciones con el objetivo de clasificar el caso anterior como "Anomalía Leve".

            # Lógica para caída rápida (pero no extremadamente) Y codo bajo
            if length_steep_slope <= THRESHOLD_LENGTH_STEEP_GOOD_VS_ANOMALY and is_elbow_score_below_threshold:
                if is_concentrated_shape:
                    return f"Buena (Pico en rango intermedio, caída muy rápida y concentrada - codo temprano y score bajo)"
                else:
                    return f"Buena (Pico en rango intermedio, caída muy rápida, pero con perfil distribuido - codo temprano y score bajo)"
            else:
                # Este caso incluye aquellos con caída rápida pero score en el codo no lo suficientemente bajo.
                return f"Anomalía Leve (Pico en rango intermedio, caída {shape_type_adjective_profile} y sostenida)"
    
    # Fallback para cualquier caso no cubierto
    return f"No clasificado (Revisar umbrales y lógica, perfil {shape_type_adjective_profile})"


# Clasificar el resultado del análisis
classification = classify_anomaly_type(analysis_results)
print(f"\nClasificación Automática: {classification}")




exit()









# Aplicar K-Means con 2 clusters a sorted_patch_anomaly_scores
num_clusters_kmeans_2 = 2
kmeans_2 = KMeans(n_clusters=num_clusters_kmeans_2, random_state=42)
kmeans_2_labels = kmeans_2.fit_predict(sorted_patch_anomaly_scores.reshape(-1, 1))

# Agrupar los datos en variables según las etiquetas de los clusters
cluster_0_data = sorted_patch_anomaly_scores[kmeans_2_labels == 0]
cluster_1_data = sorted_patch_anomaly_scores[kmeans_2_labels == 1]

# Visualizar los clusters
plt.figure(figsize=(10, 6))
plt.scatter(range(len(cluster_0_data)), cluster_0_data, label='Cluster 0', color='blue')
plt.scatter(range(len(cluster_0_data), len(cluster_0_data) + len(cluster_1_data)), cluster_1_data, label='Cluster 1', color='orange')
plt.title("K-Means Clustering (k=2) en sorted_patch_anomaly_scores")
plt.xlabel("Índice")
plt.ylabel("Puntuación de Anomalía")
plt.legend()
plt.tight_layout()

# Guardar el plot en un archivo
output_kmeans_2_plot_filename = os.path.join(plot_save_directory_on_server, 'kmeans_2_sorted_patch_anomaly_scores.png')
plt.savefig(output_kmeans_2_plot_filename)
print(f"Plot de K-Means con 2 clusters guardado en: {output_kmeans_2_plot_filename}")
plt.close()

# Imprimir los datos agrupados como vectores
print("Cluster 0 Data:", cluster_0_data)
print("Cluster 1 Data:", cluster_1_data)
# Calcular RMS (Root Mean Square) para Cluster 0
if len(cluster_0_data) > 0:
    rms_cluster_0 = np.sqrt(np.mean(np.square(cluster_0_data)))
    print(f"RMS para Cluster 0: {rms_cluster_0:.4f}")
else:
    rms_cluster_0 = 0
    print("Cluster 0 está vacío. RMS establecido en 0.")

# Calcular AVG (Average) para Cluster 1
if len(cluster_1_data) > 0:
    avg_cluster_1 = np.mean(cluster_1_data)
    print(f"AVG para Cluster 1: {avg_cluster_1:.4f}")
else:
    avg_cluster_1 = 0
    print("Cluster 1 está vacío. AVG establecido en 0.")

# Calcular la diferencia entre RMS de Cluster 0 y AVG de Cluster 1
difference = rms_cluster_0 - avg_cluster_1
print(f"Diferencia (RMS Cluster 0 - AVG Cluster 1): {difference:.4f}")


print("Sorted patch anomaly scores (descending):")
print(sorted_patch_anomaly_scores)

q_score = np.mean(top_anomaly_scores)
print(f"Q-score (promedio del 1% superior de distancias): {q_score:.4f}")

# Determinar si hay anomalía estructural basada en el Q-score
anomalia_estructural = q_score > 0.27 #aqui es o no Anomalia estructural
print(f"Anomalía estructural: {'Sí' if anomalia_estructural else 'No'}")


# 4. Visualización del mapa de calor
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(query_img_pil)
plt.title('Imagen de Consulta Original')
plt.axis('off')

plt.subplot(1, 2, 2)
# Usar 'viridis' o 'jet' son buenas opciones para heatmaps
plt.imshow(anomaly_map_final, cmap='jet')
plt.title(f'Mapa de Anomalía (Q-score: {q_score:.2f})')
plt.colorbar(label='Puntuación de Anomalía Normalizada')
plt.axis('off')

heatmap_output_filename = os.path.join(plot_save_directory_on_server, 'anomaly_heatmap_hole_000.png')
plt.tight_layout()
plt.savefig(heatmap_output_filename)
print(f"Mapa de calor de anomalías guardado en: {heatmap_output_filename}")
plt.close()

print("\n--- ¡Generación del mapa de calor y q-score completada! ---")


end_time_heatmap = time.time()
print(f"Tiempo para generar el mapa de calor: {end_time_heatmap - start_time_heatmap:.4f} segundos")
exit( )
# Salir del script si no se detecta anomalía estructural
# if not anomalia_estructural:
#     print("No se detectó anomalía estructural. Saliendo del script.")
#     exit()

# Visualización de regiones en la imagen de consulta
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import measure # Para componentes conectados y propiedades de región
import matplotlib.patches as patches # Para dibujar rectángulos
import time # Asegúrate de importar time

# --- Variables de configuración y resultados previos (ejemplo, asegúrate de que existen en tu script principal) ---
# anomaly_map_final (el heatmap normalizado de 0 a 1)
# query_img_pil (la imagen original PIL)
# plot_save_directory_on_server (directorio donde guardar los plots)
# input_size (el tamaño al que se escaló la imagen para DINOv2 y el heatmap, ej. 224)
# q_score (la puntuación general de la imagen)
# patch_anomaly_scores (array 1D con las distancias de anomalía por parche)
# H_prime, W_prime (dimensiones del mapa de características LR, ej. 16, 16)
# BACKBONE_PATCH_SIZE (ej. 14 para DINOv2)


# --------------------------------------------------------------------------------------
# Tiempo total del proceso de detección y visualización de regiones
start_time_all_plotting = time.time()
# --------------------------------------------------------------------------------------


# --- Detección y visualización de regiones de anomalía "fuertes" (con rectángulos) ---

start_time_calc_strong_regions = time.time() # Inicio del temporizador para el cálculo de regiones fuertes

strong_anomaly_region_threshold = 0.75 # Ajusta este valor para definir qué es una anomalía "fuerte"
min_region_pixel_area = 50 # Tamaño mínimo de píxeles para considerar una región (en la resolución del heatmap)

binary_strong_anomaly_map = anomaly_map_final > strong_anomaly_region_threshold

detected_strong_anomaly_regions = [] # Lista para guardar los bounding boxes de las regiones detectadas

# Calcula factores de escala (necesarios si no los tienes calculados previamente en el script principal)
original_img_width, original_img_height = query_img_pil.size
scale_x = original_img_width / input_size
scale_y = original_img_height / input_size

if np.any(binary_strong_anomaly_map):
    labeled_anomaly_regions = measure.label(binary_strong_anomaly_map)
    region_properties = measure.regionprops(labeled_anomaly_regions)
    
    for region in region_properties:
        if region.area >= min_region_pixel_area:
            # bbox de skimage es (min_row, min_col, max_row, max_col)
            min_y_at_input_size, min_x_at_input_size, max_y_at_input_size, max_x_at_input_size = region.bbox

            scaled_min_x = int(min_x_at_input_size * scale_x)
            scaled_min_y = int(min_y_at_input_size * scale_y)
            scaled_max_x = int(max_x_at_input_size * scale_x)
            scaled_max_y = int(max_y_at_input_size * scale_y)

            # Guardar el bounding box como (x_start, y_start, width, height) para matplotlib.patches.Rectangle
            region_width = scaled_max_x - scaled_min_x
            region_height = scaled_max_y - scaled_min_y

            detected_strong_anomaly_regions.append({
                'bbox': (scaled_min_x, scaled_min_y, region_width, region_height),
                'area_pixels': region.area
            })
            
end_time_calc_strong_regions = time.time() # Fin del temporizador para el cálculo de regiones fuertes
print(f"Tiempo de cálculo de regiones fuertes: {end_time_calc_strong_regions - start_time_calc_strong_regions:.4f} segundos")
# Imprimir las regiones detectadas
print(f"Shape de detected_strong_anomaly_regions: {len(detected_strong_anomaly_regions)}")
print(f"Tipo de detected_strong_anomaly_regions: {type(detected_strong_anomaly_regions)}")
print("\n--- Regiones Fuertes de Anomalía Detectadas (Vector Completo) ---")
print(detected_strong_anomaly_regions)
print("\n--- Regiones Fuertes de Anomalía Detectadas ---")
for idx, region in enumerate(detected_strong_anomaly_regions):
    print(f"Región {idx + 1}:")
    print(f"  - Bounding Box (x, y, width, height): {region['bbox']}")
    print(f"  - Área en píxeles: {region['area_pixels']}")
    
    

# Plotting de regiones fuertes (con rectángulos)
if len(detected_strong_anomaly_regions) > 0:
    plt.figure(figsize=(10, 8))
    plt.imshow(query_img_pil)
    plt.title(f'Imagen de Consulta con Regiones Fuertes de Anomalía (Q-score: {q_score:.2f})')
    plt.axis('off')

    ax = plt.gca() # Obtener el eje actual de matplotlib
    for region_info in detected_strong_anomaly_regions:
        bbox = region_info['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=3, edgecolor='lime', facecolor='none',
                                 linestyle='-', alpha=0.9) # Rectángulos verdes sólidos
        ax.add_patch(rect)
    
    # Leyenda para los rectángulos
    ax.add_patch(patches.Rectangle((0,0), 0.1, 0.1, linewidth=3, edgecolor='lime', facecolor='none', linestyle='-', alpha=0.9, label=f'Regiones Fuertes de Anomalía'))
    plt.legend()

    strong_regions_overlay_output_filename = os.path.join(plot_save_directory_on_server, 'strong_anomaly_regions_overlay_hole_000.png')
    plt.tight_layout()
    plt.savefig(strong_regions_overlay_output_filename)
    plt.close()

# --------------------------------------------------------------------------------------
end_time_all_plotting = time.time()
print(f"Tiempo total para procesar y dibujar las regiones: {end_time_all_plotting - start_time_all_plotting:.4f} segundos")
# --------------------------------------------------------------------------------------






##################################

# --- Función para buscar imágenes similares usando KNN (OPTIMIZADA) ---
def buscar_imagenes_similares_knn(query_feature_map, pre_flattened_features_bank, k=3, nombres_archivos=None):
    """
    Busca imágenes similares usando KNN.
    `pre_flattened_features_bank` es un array NumPy (o tensor PyTorch) ya aplanado y apilado.
    `nombres_archivos` es una lista de nombres de archivos en el mismo orden.
    """
    query_feat_flatten = query_feature_map.flatten().cpu().numpy()

    # El banco de características ya está aplanado y apilado
    features_bank_for_knn = pre_flattened_features_bank
    
    # Asegurarse de que el banco de características esté en NumPy para sklearn
    if isinstance(features_bank_for_knn, torch.Tensor):
        features_bank_for_knn = features_bank_for_knn.cpu().numpy()

    # Medir el tiempo de cálculo de distancias KNN
    start_time_knn_dist = time.time()
    distances = euclidean_distances([query_feat_flatten], features_bank_for_knn)
    nearest_indices = np.argsort(distances[0])[:k]
    end_time_knn_dist = time.time()
    time_knn_dist = end_time_knn_dist - start_time_knn_dist
    print(f"Tiempo para calcular distancias KNN: {end_time_knn_dist - start_time_knn_dist:.4f} segundos")

    imagenes_similares = []
    rutas_imagenes_similares = []
    if nombres_archivos:
        for i in nearest_indices:
            imagenes_similares.append(nombres_archivos[i])
            rutas_imagenes_similares.append(os.path.join(directorio_imagenes, nombres_archivos[i]))
    else: # Fallback si no se proporcionan nombres
        for i in nearest_indices:
            imagenes_similares.append(f"Imagen_Banco_{i:03d}.png") # Nombre genérico
            rutas_imagenes_similares.append(os.path.join(directorio_imagenes, f"Imagen_Banco_{i:03d}.png")) # Puede que no exista

    print(f"Índices de los {k} vecinos más cercanos: {nearest_indices}")
    print("Imágenes similares:", imagenes_similares)
    return imagenes_similares, rutas_imagenes_similares, time_knn_dist

# --- Uso de la función de búsqueda KNN con el banco de características pre-aplanado ---
print("\nRealizando búsqueda KNN de imágenes similares usando el banco pre-aplanado del Coreset...")
# Ahora pasamos 'coreset_relevant_flat_features_bank' y 'coreset_relevant_filenames'
imagenes_similares, rutas_imagenes_similares, time_knn_dist = buscar_imagenes_similares_knn(
    query_lr_features, coreset_relevant_flat_features_bank, nombres_archivos=coreset_relevant_filenames
)

print("\n--- Resultados Finales de la Búsqueda KNN ---")
print("Imágenes similares encontradas:", imagenes_similares)
print("Rutas de imágenes similares:", rutas_imagenes_similares)

# --- Visualización de las imágenes similares ---
print("\nVisualizando las imágenes similares encontradas...")
plt.figure(figsize=(15, 5))
plt.subplot(1, len(rutas_imagenes_similares) + 1, 1) # Ajuste para incluir la imagen de consulta
plt.imshow(query_img_pil)
plt.title('Imagen de Consulta')
plt.axis('off')

for i, ruta_imagen_similar in enumerate(rutas_imagenes_similares):
    try:
        img_similar = Image.open(ruta_imagen_similar).convert('RGB')
        plt.subplot(1, len(rutas_imagenes_similares) + 1, i + 2) # i + 2 porque la primera es la de consulta
        plt.imshow(img_similar)
        plt.title(f'Vecino {i + 1}\n({os.path.basename(ruta_imagen_similar)})')
        plt.axis('off')
    except FileNotFoundError:
        print(f"Advertencia: No se pudo encontrar la imagen en la ruta: {ruta_imagen_similar}")
        plt.subplot(1, len(rutas_imagenes_similares) + 1, i + 2)
        plt.text(0.5, 0.5, "Imagen no encontrada", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(f'Vecino {i + 1}\n(No encontrada)')
        plt.axis('off')

output_similar_plot_filename = os.path.join(plot_save_directory_on_server, 'similar_images_plot.png')
plt.tight_layout()
plt.savefig(output_similar_plot_filename)
print(f"Plot de imágenes similares guardado en: {output_similar_plot_filename}")
plt.close()
print("Script Stage 2 completado.")

# --- 6. Aplicar FeatUp para obtener características de alta resolución ---

def apply_featup_hr(image_path, featup_upsampler, image_transform, device):
    """
    Aplica FeatUp para obtener características de alta resolución y sus LR correspondientes.
    Usa la misma transformación para la entrada del backbone y la guía.
    """
    image_tensor = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        hr_feats = featup_upsampler(image_tensor) # image_tensor se usa como input Y como guidance
        lr_feats = featup_upsampler.model(image_tensor) # Las características del backbone
    return lr_feats.cpu(), hr_feats.cpu() # Devolvemos a CPU

#Area de mejora: no calcular otra vez los lr_feats, sino emplear los almacenados en el Banco. Es mas eficiente?

# 6.1 A imagen de consulta
start_time = time.time()
input_query_tensor = transform(Image.open(query_image_path).convert("RGB")).unsqueeze(0).to(device)
query_lr_feats_featup, query_hr_feats = apply_featup_hr(query_image_path, upsampler, transform, device)
print(f"Dimensiones de características de referencia (alta resolución): {query_lr_feats_featup.shape}")
print(f"Dimensiones de características de referencia (alta resolución): {query_hr_feats.shape}")
end_time = time.time()
plot_feats(unnorm(input_query_tensor)[0], query_lr_feats_featup[0], query_hr_feats[0])
# Guardar el plot en un archivo
output_query_plot_filename = os.path.join(plot_save_directory_on_server, 'query_image_features_plot.png')
plt.tight_layout()
plt.savefig(output_query_plot_filename)
print(f"Plot de características de la imagen de consulta guardado en: {output_query_plot_filename}")
plt.close()

# 6.2 A similares imagenes
similar_hr_feats_list = []
print("Imágenes similares:", imagenes_similares)
for i, similar_image_path in enumerate(rutas_imagenes_similares):  # Usa los paths
    # Cargar y transformar la imagen similar
    input_similar_tensor = transform(Image.open(similar_image_path).convert("RGB")).unsqueeze(0).to(device)
    
    # Aplicar FeatUp para obtener características de baja y alta resolución
    similar_lr_feats, similar_hr_feats = apply_featup_hr(similar_image_path, upsampler, transform, device)
    similar_hr_feats_list.append(similar_hr_feats)
    # Visualizar las características de la imagen similar
    plt.figure(figsize=(10, 5))  # Ajusta el tamaño del plot si es necesario
    plot_feats(unnorm(input_similar_tensor)[0], similar_lr_feats[0], similar_hr_feats[0])
    
    # Guardar el plot en un archivo
    output_similar_plot_filename = os.path.join(plot_save_directory_on_server, f'similar_image_{i + 1}_features_plot.png')
    plt.tight_layout()
    plt.savefig(output_similar_plot_filename)
    print(f"Plot de características de la imagen similar guardado en: {output_similar_plot_filename}")
    plt.close()  # Liberar memoria después de guardar el plot

########################
# APLICANDO SAM MASK

print(f"iniciando SAm")
start_time_sam = time.time()
# Importamos SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2

# Funciones de visualización (las mismas que en tu código original)
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    
    # Create a 3-channel image for drawing contours
    # Initialize a transparent image for the mask itself
    mask_image_alpha = np.zeros((h, w, 4), dtype=np.float32)
    mask_image_alpha[mask > 0] = color

    if borders:
        # Convert mask to uint8 for findContours
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Draw contours on a separate 3-channel image first, then blend
        # Create a blank 3-channel image to draw contours on
        contour_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # The contour color for cv2.drawContours should be 3 channels (BGR)
        # Convert (1,1,1) (white) to (255,255,255) for uint8 image
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=2)
        
        # Convert contour image to float and normalize to [0,1] for blending
        contour_image_float = contour_image.astype(np.float32) / 255.0
        
        # Blend the contours onto the mask_image_alpha
        # Create a mask for the contours themselves to apply their color
        contour_mask = (contour_image_float.sum(axis=-1) > 0).astype(np.float32)
        
        # Apply white color to the contour regions in the alpha image
        # This will make the borders white on the final plot
        mask_image_alpha[contour_mask > 0, :3] = 1.0 # Set RGB to white
        mask_image_alpha[contour_mask > 0, 3] = 0.5 # Set alpha for the border

    ax.imshow(mask_image_alpha)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_masks_grid(image, masks, points=None, plot_title="Generated Masks"):
    """
    Función adaptada para mostrar múltiples máscaras de un grid de puntos.
    Muestra todas las máscaras generadas en una sola figura.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    if points is not None:
        # Los puntos del grid generados automáticamente por SAM2AutomaticMaskGenerator
        # son siempre positivos (label=1). No necesitamos mostrar negativos aquí.
        # Creamos labels para visualización.
        point_labels_for_display = np.ones(points.shape[0], dtype=int)
        show_points(points, point_labels_for_display, ax, marker_size=50) # Reducimos el tamaño para un grid

    for mask_data in masks:
        mask = mask_data["segmentation"]
        show_mask(mask, ax, random_color=True) # Usa colores aleatorios para distinguir máscaras
    
    plt.title(plot_title, fontsize=18)
    plt.axis('off')
    # Guardar el plot en un archivo en lugar de mostrarlo interactivamente
    #output_grid_mask_filename = os.path.join(plot_save_directory_on_server, 'sam_grid_masks.png')
    #plt.savefig(output_grid_mask_filename)
    #print(f"Plot de máscaras del grid guardado en: {output_grid_mask_filename}")
    #plt.close()


checkpoint = "/home/imercatoma/sam2_repo_independent/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

try:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=True) # apply_postprocessing=False para AutomaticMaskGenerator
    print("SAM2 cargado correctamente para Automatic Mask Generation")
except FileNotFoundError as e:
    print(f"Error: No se encontraron los archivos del modelo SAM2: {e}")
    exit()
except Exception as e:
    print("Error al cargar SAM2:")
    print(e)
    exit()

# Cargar la imagen de consulta para SAM
try:
    image_for_sam = Image.open(query_image_path).convert("RGB")
    image_for_sam_np = np.array(image_for_sam)
    print(f"Dimensiones imagen de entrada a SAM (np.array(query_img_pil)): {image_for_sam_np.shape}")
except FileNotFoundError:
    print(f"Error: No se encontró la imagen en la ruta: {query_image_path}")
    exit()

# --- Parámetros para el grid de puntos y filtrado de máscaras ---
points_grid_density = 16 # 16 - 12 etc Número de puntos a lo largo de un lado del grid
min_mask_area_pixels = 100.0 #  100 equivale a 1000 en mask_info Área mínima de la máscara para filtrar (en píxeles)
max_mask_area_pixels = 450000.0 # Área máxima de la máscara para filtrar (en píxeles)

# Inicializar el generador de máscaras automático
mask_generator_query = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=points_grid_density, # Usamos la variable que definimos
    points_per_batch=256, # 64 - 256 Número de puntos procesados en cada lote (ajustable según tu GPU)
    pred_iou_thresh=0.48, # 0.4  0.88 Umbral de confianza para filtrar máscaras
    stability_score_thresh=0.7, # 0.8 Umbral de estabilidad para filtrar máscaras
    crop_n_layers=0, # Desactiva el cropping batch de recorte 
    #crop_n_points_downscale_factor (por defecto 1) depende de crop_n_layers > 1
    min_mask_region_area=min_mask_area_pixels, # Área mínima de la máscara para filtrar (en píxeles)
)

# Usamos la variable 'points_grid_density' que contiene el valor
print(f"Generando máscaras con un grid de {points_grid_density}x{points_grid_density} puntos...")

# Generar máscaras
masks_data = mask_generator_query.generate(image_for_sam_np)
print(f"Tipo de dato de masks_data: {type(masks_data)}")

# Filtra las máscaras directamente y reasigna el resultado a masks_data
masks_data = [mask_info for mask_info in masks_data if mask_info['area'] <= max_mask_area_pixels]
print(f"Se generaron {len(masks_data)} máscaras después de filtrar por área máxima.")
# El resto de tu código que utiliza 'masks_data' ahora operará sobre las máscaras filtradas.



print(f"Se generaron {len(masks_data)} máscaras de consulta.")
#print(f"Dimensiones mascaras: {masks_data.shape}")
print(f"Dimensiones de la imagen de entrada a SAM: {image_for_sam_np.shape}")
# Mostrar información de las máscaras generadas

for i, mask_info in enumerate(masks_data):
    print(f"Mascara {i + 1}:")
    print(f"  - Dimensiones: {mask_info['segmentation'].shape}")
    print(f"  - Área: {mask_info['area']}")
    print(f"  - Puntos usados: {mask_info.get('point_coords', 'N/A')}")
    print(f"  - Etiquetas de puntos: {mask_info.get('point_labels', 'N/A')}")
    print(f"  - Predicción de IoU: {mask_info.get('predicted_iou', 'N/A')}")
    print(f"  - Estabilidad: {mask_info.get('stability_score', 'N/A')}\n")

# --- Visualización de las máscaras generadas ---

# Extraer los puntos que SAM2AutomaticMaskGenerator usó para generar las máscaras
# SAM2AutomaticMaskGenerator no retorna directamente el grid completo,
# sino los puntos que generaron cada máscara. Podemos colectarlos para visualización.
all_generated_points = []
for mask_info in masks_data:
    if "point_coords" in mask_info:
        all_generated_points.append(mask_info["point_coords"])
if all_generated_points:
    all_generated_points = np.concatenate(all_generated_points, axis=0)
else:
    all_generated_points = None

# Visualizar las máscaras generadas y los puntos del grid
show_masks_grid(image_for_sam_np, masks_data, points=all_generated_points, plot_title=f"SAM2 Masks with {points_grid_density}x{points_grid_density} Grid Points")
# --- Save the query image mask plot with a unique filename ---
output_query_grid_mask_filename = os.path.join(plot_save_directory_on_server, 'sam_query_image_grid_masks.png')
plt.savefig(output_query_grid_mask_filename)
print(f"Plot de máscaras del grid para la imagen de consulta guardado en: {output_query_grid_mask_filename}")
plt.close() # Close the figure to free memory

# --------------Aplicando SAM MASK a las imágenes similares ---
# Inicializar el generador de máscaras automático similares
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=points_grid_density, # Usamos la variable que definimos
    points_per_batch=256, # 64 - 256 Número de puntos procesados en cada lote (ajustable según tu GPU)
    pred_iou_thresh=0.48, # 0.4  0.88 Umbral de confianza para filtrar máscaras
    stability_score_thresh=0.7, # 0.8 Umbral de estabilidad para filtrar máscaras
    crop_n_layers=0, # Desactiva el cropping batch de recorte 
    #crop_n_points_downscale_factor (por defecto 1) depende de crop_n_layers > 1
    min_mask_region_area=800, # Área mínima de la máscara para filtrar (en píxeles)
)


# --- Aplicando SAM MASK a las imágenes similares ---
print("\nGenerando y visualizando máscaras SAM para las imágenes similares...")
similar_masks_raw_list = []
for i, similar_image_path in enumerate(rutas_imagenes_similares):
    try:
        img_similar_pil = Image.open(similar_image_path).convert('RGB')
        image_np_similar_for_sam = np.array(img_similar_pil)

        print(f"--- Procesando: {os.path.basename(similar_image_path)} ---") # Add a clear separator
        current_similar_masks_data = mask_generator.generate(image_np_similar_for_sam)
        print(f"Se generaron {len(current_similar_masks_data)} máscaras para la imagen similar {i + 1}.")
        
        # Filtrar por área máxima
        current_similar_masks_data = [
            mask_info for mask_info in current_similar_masks_data 
            if mask_info['area'] <= max_mask_area_pixels
        ]
        
        similar_masks_raw_list.append(current_similar_masks_data) # Guardar las máscaras generadas para esta imagen
        print(f"Dimensiones imagen similar para SAM (np.array): {image_np_similar_for_sam.shape}")
        #print(f"Se generaron {len(current_similar_masks_data)} máscaras para {os.path.basename(similar_image_path)}.")

        all_generated_points_similar = []
        for mask_info in current_similar_masks_data:
            if "point_coords" in mask_info:
                all_generated_points_similar.append(mask_info["point_coords"])
        if all_generated_points_similar:
            all_generated_points_similar = np.concatenate(all_generated_points_similar, axis=0)
        else:
            all_generated_points_similar = None

        # Call show_masks_grid (it no longer saves the file internally)
        show_masks_grid(image_np_similar_for_sam, current_similar_masks_data, 
                        points=all_generated_points_similar, 
                        plot_title=f"SAM2 Masks - Vecino {i + 1} ({os.path.basename(similar_image_path)})")
        
        # --- NOW, SAVE WITH A UNIQUE FILENAME FOR THIS SIMILAR IMAGE ---
        output_similar_grid_mask_filename = os.path.join(plot_save_directory_on_server, f'sam_similar_image_{i + 1}_grid_masks.png')
        plt.savefig(output_similar_grid_mask_filename) # Save the plot for this image
        print(f"Plot de máscaras del grid para el vecino {i + 1} guardado en: {output_similar_grid_mask_filename}")
        plt.close() # Close the figure after saving and for the next iteration

    except FileNotFoundError:
        print(f"Advertencia: No se pudo encontrar la imagen similar en la ruta: {similar_image_path}. Omitiendo generación de máscaras para esta imagen.")
    except Exception as e:
        print(f"Error al procesar la imagen similar {os.path.basename(similar_image_path)} para la generación de máscaras SAM: {e}")


end_time_sam = time.time()
print(f"Tiempo de ejecución de SAM con grid: {end_time_sam - start_time_sam:.4f} segundos")



# --- Implementación del punto 3.4.3. Object Feature Map ---
import torch.nn.functional as F # Importa F para F.interpolate

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
        # Reshape the mask to match the original image size that SAM processed (e.g., 1024x1024)
        # This is already handled by SAM's output, but explicitly stating its shape
        # is good for clarity.
        mask_tensor_original_res = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0) # (1, 1, H_orig, W_orig)

        # Mover la máscara al mismo dispositivo que el mapa de características HR
        mask_tensor_original_res = mask_tensor_original_res.to(hr_feature_map.device)

        # 1. Escalar la máscara a (8H', 8W') usando interpolación bilineal
        # La interpolación se realiza desde la resolución original de SAM (ej. 1024x1024)
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

# --- Aplicar el proceso a la imagen de consulta y a las imágenes de referencia ---

print("\n--- Generando Mapas de Características de Objeto ---")


# Dimensiones objetivo para las máscaras después de escalar (8H', 8W')
TARGET_MASK_H = 8 * H_prime # 8 * 16 = 128
TARGET_MASK_W = 8 * W_prime # 8 * 16 = 128
# Para la imagen de consulta (Iq)
fobj_q = process_masks_to_object_feature_maps(
    masks_data, #query_masks_raw,
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

print("\nProceso de 'Object Feature Map' completado. ¡Ahora tienes los fobj_q y fobj_r listos!")
  


# -----------3.5.2 Object matching module----------------------------------------------------------------
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

# --- Funciones de ploteo MODIFICADAS para la matriz P y P_augmented_full ---
def plot_assignment_matrix(P_matrix, query_labels, reference_labels, save_path=None, title="Matriz de Asignación P"):
    """
    Visualiza la matriz de asignación P como un mapa de calor y muestra los coeficientes
    de confianza dentro de cada celda.

    Args:
        P_matrix (torch.Tensor or np.array): La matriz de asignación (M x N).
        query_labels (list): Etiquetas para los objetos de consulta (eje Y).
        reference_labels (list): Etiquetas para los objetos de referencia (eje X).
        save_path (str, optional): Ruta para guardar la imagen del plot.
        title (str): Título del plot.
    """
    if isinstance(P_matrix, torch.Tensor):
        P_matrix = P_matrix.detach().cpu().numpy()

    plt.figure(figsize=(P_matrix.shape[1] * 1.0 + 2, P_matrix.shape[0] * 1.0 + 2)) # Ajustado figsize
    plt.imshow(P_matrix, cmap='viridis', origin='upper', aspect='auto')
    plt.colorbar(label='Probabilidad de Asignación')
    plt.xticks(np.arange(len(reference_labels)), reference_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(query_labels)), query_labels)
    plt.xlabel('Objetos de Referencia')
    plt.ylabel('Objetos de Consulta')
    plt.title(title)

    # Añadir los valores de confianza dentro de cada celda
    for i in range(P_matrix.shape[0]):
        for j in range(P_matrix.shape[1]):
            # Determinar el color del texto basado en el fondo (valor de la celda)
            # Esto ayuda a la legibilidad: texto blanco sobre fondo oscuro, texto negro sobre fondo claro
            text_color = 'white' if P_matrix[i, j] < 0.5 else 'black' 
            plt.text(j, i, f'{P_matrix[i, j]:.3f}',
                     ha="center", va="center", color=text_color, fontsize=8,
                     weight='bold' if P_matrix[i, j] > 0.5 else 'normal') # Negrita para valores altos


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot de la matriz de asignación guardado en: {save_path}")
    plt.show()
    plt.close()

def plot_augmented_assignment_matrix(P_augmented_full, query_labels, reference_labels, save_path=None, title="Matriz de Asignación Aumentada (con Trash Bin)"):
    """
    Visualiza la matriz de asignación aumentada (incluyendo los trash bins) como un mapa de calor
    y muestra los coeficientes de confianza dentro de cada celda.

    Args:
        P_augmented_full (torch.Tensor or np.array): La matriz de asignación aumentada ((M+1) x (N+1)).
        query_labels (list): Etiquetas para los objetos de consulta.
        reference_labels (list): Etiquetas para los objetos de referencia.
        save_path (str, optional): Ruta para guardar la imagen del plot.
        title (str): Título del plot.
    """
    if isinstance(P_augmented_full, torch.Tensor):
        P_augmented_full = P_augmented_full.detach().cpu().numpy()

    # Añadir etiquetas para los trash bins
    full_query_labels = [f"Q_{i}" for i in query_labels] + ["Trash Bin (Q)"]
    full_reference_labels = [f"R_{i}" for i in reference_labels] + ["Trash Bin (R)"]

    plt.figure(figsize=(P_augmented_full.shape[1] * 1.0 + 2, P_augmented_full.shape[0] * 1.0 + 2)) # Ajustado figsize
    plt.imshow(P_augmented_full, cmap='viridis', origin='upper', aspect='auto')
    plt.colorbar(label='Probabilidad de Asignación')
    plt.xticks(np.arange(len(full_reference_labels)), full_reference_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(full_query_labels)), full_query_labels)
    plt.xlabel('Objetos de Referencia y Trash Bin')
    plt.ylabel('Objetos de Consulta y Trash Bin')
    plt.title(title)

    # Añadir los valores de confianza dentro de cada celda
    for i in range(P_augmented_full.shape[0]):
        for j in range(P_augmented_full.shape[1]):
            text_color = 'white' if P_augmented_full[i, j] < 0.5 else 'black'
            plt.text(j, i, f'{P_augmented_full[i, j]:.3f}',
                     ha="center", va="center", color=text_color, fontsize=8,
                     weight='bold' if P_augmented_full[i, j] > 0.5 else 'normal')


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot de la matriz de asignación aumentada guardado en: {save_path}")
    plt.show()
    plt.close()

# --- Fin de las funciones de ploteo MODIFICADAS ---

## Matching-continue---
## Matching
start_time_sam_matching = time.time()
def apply_global_max_pool(feat_map):
    return F.adaptive_max_pool2d(feat_map, output_size=1).squeeze(-1).squeeze(-1)

fobj_q_pooled = apply_global_max_pool(fobj_q)

all_fobj_r_pooled_list = []
for fobj_r_current in all_fobj_r_list:
    pooled_r = apply_global_max_pool(fobj_r_current)
    all_fobj_r_pooled_list.append(pooled_r)

fobj_q_norm = F.normalize(fobj_q_pooled, p=2, dim=1) # --> d_M_q tensor (M, C)

all_fobj_r_norm_list = [F.normalize(fobj_r_pooled, p=2, dim=1)
                        for fobj_r_pooled in all_fobj_r_pooled_list] # --> d_N_r tensor (N, C)

def max_similarities(query_feats, candidate_feats):
    sim_matrix = torch.mm(query_feats, candidate_feats.T)# [-1, 1] producto punto...similar a similitud coseno si se normalizan A y B a vectores unitarios
    max_vals, _ = sim_matrix.max(dim=1)
    return max_vals

# --- Optimal Matching Module ---
class ObjectMatchingModule(nn.Module):
    def __init__(self, superglue_weights_path=None, sinkhorn_iterations=100, sinkhorn_epsilon=0.1, custom_z=None):
        super(ObjectMatchingModule, self).__init__()
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon

        if custom_z is not None: # Usar custom_z si se proporciona
            self.z = nn.Parameter(torch.tensor(custom_z, dtype=torch.float32))
            print(f"Parámetro 'z' inicializado con valor personalizado: {self.z.item():.4f}")
        elif superglue_weights_path and os.path.exists(superglue_weights_path):
            try:
                state_dict = torch.load(superglue_weights_path, map_location=device)
                if 'bin_score' in state_dict:
                    z_value = state_dict['bin_score'].item()
                elif 'match_model.bin_score' in state_dict:
                    z_value = state_dict['match_model.bin_score'].item()
                else:
                    print(f"Advertencia: 'z' (bin_score) no encontrado en {superglue_weights_path}. Inicializando con valor predeterminado.")
                    z_value = 0.5
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
        # print(f"Matriz de similitud inicial (MxN): {score_matrix.shape}")
        # print("score_matrix:\n", score_matrix) # Descomentar para ver la matriz

        S_augmented = torch.zeros((M + 1, N + 1), device=d_M_q.device, dtype=d_M_q.dtype)
        S_augmented[:M, :N] = score_matrix
        S_augmented[:M, N] = self.z # Última columna (trash bin para query)
        S_augmented[M, :N] = self.z # Última fila (trash bin para reference)
        S_augmented[M, N] = self.z # Esquina inferior derecha (trash bin vs trash bin)
        # print(f"Matriz S_augmented (M+1 x N+1): {S_augmented.shape}")
        # print("S_augmented:\n", S_augmented) # Descomentar para ver la matriz

        K = torch.exp(S_augmented / self.sinkhorn_epsilon)
        print("K (exp(S/epsilon)) antes de Sinkhorn:\n", K) # Descomentar para ver

        for i in range(self.sinkhorn_iterations):
            K = K / K.sum(dim=1, keepdim=True) # Normalizar filas
            K = K / K.sum(dim=0, keepdim=True) # Normalizar columnas
            print(f"K después de iteración {i+1} de Sinkhorn:\n", K) # Descomentar para ver cada iteración

        P_augmented_full = K
        P = P_augmented_full[:M, :N] # La matriz de asignación sin trash bins
        # print(f"Matriz de asignación P (MxN): {P.shape}")
        # print("P:\n", P) # Print de la matriz de asignación P
        # print(f"Matriz de asignación P_augmented_full (M+1 x N+1): {P_augmented_full.shape}")
        # print("P_augmented_full:\n", P_augmented_full) # Print de la matriz de asignación completa

        return P, P_augmented_full

# --- Uso del módulo de coincidencia ---
superglue_weights_path = "/home/imercatoma/superglue_indoor.pth"#
# Para probar el ajuste de `self.z`, puedes pasar `custom_z` aquí:
# Prueba con valores como 0.0, 0.1, 0.2, etc.
object_matching_module = ObjectMatchingModule(
    superglue_weights_path=superglue_weights_path,
    sinkhorn_iterations=30,  #100
    sinkhorn_epsilon=0.1,
    custom_z=0.9 #0.9 # <--- ¡Modifica este valor para tus pruebas! 1.5 a 4 funciona como 2.32 de superglue 1.1 ya varia 0.8
    # bajar eleva el trash bin, subir lo baja [0.6 -- trashR 0.35] mal resultado
).to(device)


P_matrices = []
P_augmented_full_matrices = []
# Obtener etiquetas para los plots de la matriz
query_obj_labels = [f"obj_{i}" for i in range(fobj_q_norm.shape[0])]

for i, d_N_r_current_image in enumerate(all_fobj_r_norm_list): # Cambié el nombre para mayor claridad
    d_M_q = fobj_q_norm.to(device) # Aseguramos que fobj_q_norm esté en el dispositivo correcto
    d_N_r_current_image = d_N_r_current_image.to(device)

    print(f"\n--- Procesando coincidencia para imagen de referencia {i+1} ---")
    P_current, P_augmented_current = object_matching_module(d_M_q, d_N_r_current_image)
    P_matrices.append(P_current)
    P_augmented_full_matrices.append(P_augmented_current)
    print(f"Coincidencia para imagen de referencia {i+1} completada.")

    # Generar etiquetas para los objetos de referencia de la imagen actual
    current_ref_obj_labels = [f"obj_{j}" for j in range(d_N_r_current_image.shape[0])]

    # Plotear la matriz P (sin trash bin)
    output_p_matrix_filename = os.path.join(plot_save_directory_on_server, f'assignment_matrix_P_ref_{i+1}.png')
    plot_assignment_matrix(P_current, query_obj_labels, current_ref_obj_labels,
                           save_path=output_p_matrix_filename,
                           title=f"Matriz de Asignación (Query vs. Ref {i+1})")

    # Plotear la matriz P_augmented_full (con trash bin)
    output_p_augmented_filename = os.path.join(plot_save_directory_on_server, f'augmented_assignment_matrix_ref_{i+1}.png')
    plot_augmented_assignment_matrix(P_augmented_current, query_obj_labels, current_ref_obj_labels,
                                     save_path=output_p_augmented_filename,
                                     title=f"Matriz de Asignación Aumentada (Query vs. Ref {i+1})")


# --- Lógica de Detección de Anomalías ---
# fobj_q_norm:(magnitud 1) normalizado a lo largo de una dim=1: en las columnas p=2: L2 distance
M = fobj_q_norm.shape[0]#devuelve el número de filas en el tensor que es el número de objetos de consulta total
# la suma de sus cuadrados es 1.0; cada vector de características es escalado para que su norma L2 sea 1.0, vector unitario, valores individuales no restringidos a [0,1]
is_matched_to_real_object = torch.zeros(M, dtype=torch.bool, device=device)
best_match_confidence_overall = torch.full((M,), -1.0, device=device) # inicializar con -1.0 para indicar que no hay coincidencia
best_match_to_trash_bin_confidence = torch.full((M,), -1.0, device=device)# almacena la mejor confianza de coincidencia con el trash bin

anomaly_detection_threshold = 0.10 # Umbral de detección de anomalías



for P_current, P_augmented_current in zip(P_matrices, P_augmented_full_matrices):
    if P_current.shape[0] == 0: continue

    M_current, N_current = P_current.shape

    max_conf_to_real_ref, _ = P_current.max(dim=1)  #sacar el max en columnas, devuelve el máximo de cada fila (query) y su índice
    # tensor de máximos por fila (query vs. trash bin)
    #tensor: contiene max confianza de asign a cada objeto de consulta a los objetos de referencia reales

    conf_to_trash_bin_current = P_augmented_current[:M_current, N_current]#Para cada objeto de consulta, se extrae la confianza de asignación al "trash bin" desde la última columna

    for q_idx in range(M_current):
        if max_conf_to_real_ref[q_idx] > best_match_confidence_overall[q_idx]: #actualizar al maximo global best match iterando
            best_match_confidence_overall[q_idx] = max_conf_to_real_ref[q_idx]

        if conf_to_trash_bin_current[q_idx] > best_match_to_trash_bin_confidence[q_idx]: #actualizar al maximo global bin
             best_match_to_trash_bin_confidence[q_idx] = conf_to_trash_bin_current[q_idx]
                                        # anomaly_detection_threshold
        if max_conf_to_real_ref[q_idx] > anomaly_detection_threshold and \
           max_conf_to_real_ref[q_idx] > conf_to_trash_bin_current[q_idx]:
            is_matched_to_real_object[q_idx] = True

anomalies_final = ~is_matched_to_real_object

anomalous_ids = anomalies_final.nonzero(as_tuple=True)[0].tolist()
anomalous_info = []


for idx in range(M):
    if idx in anomalous_ids:
        anomalous_info.append((idx, best_match_confidence_overall[idx].item())) # Incluir la mejor similitud real
        print(f"Objeto {idx} es anómalo. Mejor similitud real: {best_match_confidence_overall[idx].item():.4f}, Confianza a trash bin: {best_match_to_trash_bin_confidence[idx].item():.4f}")
    else:
        print(f"Objeto {idx} asignado a un objeto real. Mejor similitud real: {best_match_confidence_overall[idx].item():.4f}, Confianza a trash bin: {best_match_to_trash_bin_confidence[idx].item():.4f}")


output_anomalies_query_plot_om = os.path.join(plot_save_directory_on_server, 'query_image_anomalies_optimal_good_000.png')
# Aquí también, pasamos la imagen original `image_for_sam_np`
show_anomalies_on_image(image_for_sam_np, masks_data, anomalous_info, alpha=0.5, save_path=output_anomalies_query_plot_om)
print(f"Plot de anomalías guardado en: {output_anomalies_query_plot_om}")




#------------------------------------- 3.6 Anomaly detection-------------------------------------------

# --- 3.6 Anomaly Measurement Module (AMM) ---
#------------------------------------ 3.6 Anomaly detection-------------------------------------------Add commentMore actions


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import seaborn as sns
from sklearn.cluster import KMeans

# Asumiendo que 'device' y otras variables globales están definidas en tu script principal
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TARGET_MASK_H y TARGET_MASK_W (actualmente 128x128 para el cálculo de scores)
# H_prime, W_prime (la resolución nativa de DinoV2, 16x16)
# Definiciones de calculate_anomaly_map_matched y calculate_anomaly_map_unmatched_mahalanobis
# y build_reference_pixel_distributions (las que acabas de proporcionar) van aquí.

# --- Tu función calculate_anomaly_map_matched (sin cambios en la función en sí) ---
def calculate_anomaly_map_matched(fobj_q_i, fobj_r_j, query_mask_original_shape, target_mask_h, target_mask_w):
    """
    Calculates the L2 distance anomaly map for a matched query object and its reference.
    Features are (C, H_feat, W_feat)
    """
    C, H_feat, W_feat = fobj_q_i.shape

    # Ensure features are on the same device and are 3D (C, H_feat, W_feat)
    fobj_q_i_proc = fobj_q_i.to(device)
    fobj_r_j_proc = fobj_r_j.to(device)

    # Calculate L2 distance at each feature map pixel
    # The norm is taken over the feature dimension (C)
    diff = fobj_q_i_proc - fobj_r_j_proc
    l2_distance_map = torch.norm(diff, p=2, dim=0) # Shape (H_feat, W_feat)

    # Resize the anomaly map back to the original query image mask dimensions
    # for consistent plotting and full image anomaly map creation
    # Using F.interpolate for upsampling
    l2_distance_map_resized = F.interpolate(
        l2_distance_map.unsqueeze(0).unsqueeze(0), # Add batch and channel dims
        size=(query_mask_original_shape[0], query_mask_original_shape[1]), # Original image H, W
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0) # Remove batch and channel dims

    return l2_distance_map_resized # Shape (Original_H, Original_W)


# --- Tu función calculate_anomaly_map_unmatched_mahalanobis (sin cambios en la función en sí) ---
def calculate_anomaly_map_unmatched_mahalanobis(fobj_q_i_pixel_feat, ref_pixel_features_flat):
    """
    Calculates Mahalanobis distance for a single pixel feature from an unmatched query object
    against a collection of reference pixel features.
    fobj_q_i_pixel_feat: (C,)
    ref_pixel_features_flat: (Num_ref_pixels, C)
    """
    if ref_pixel_features_flat.shape[0] < ref_pixel_features_flat.shape[1]:
        # Handle case where Num_ref_pixels < C, leading to singular covariance.
        # This can happen if there aren't enough reference objects or if all objects are very similar.
        # Add a small identity matrix to covariance for numerical stability, or fall back to L2.
        # For simplicity, for now, we'll return a large score if covariance is singular.
        # A more robust solution would be to use a pseudo-inverse or add a small regularization.
        # For now, let's use a small constant for numerical stability of inverse
        print(f"Warning: Not enough reference samples ({ref_pixel_features_flat.shape[0]}) for robust Mahalanobis calculation. Falling back to L2-like distance or handling singular matrix.")
        # Fallback to squared Euclidean distance if not enough samples for covariance
        if ref_pixel_features_flat.shape[0] == 0:
            return torch.tensor(float('inf'), device=device) # No reference data
        # Use mean for comparison if not enough samples for covariance
        mean_ref_feat = torch.mean(ref_pixel_features_flat, dim=0)
        return torch.norm(fobj_q_i_pixel_feat - mean_ref_feat, p=2) ** 2 # Squared L2 distance

    mean_ref_feat = torch.mean(ref_pixel_features_flat, dim=0) # (C,)
    
    # Calculate covariance matrix
    # torch.cov requires input as (num_features, num_samples) for columns as features.
    # Our `ref_pixel_features_flat` is (Num_samples, C), so transpose it.
    try:
        cov_matrix = torch.cov(ref_pixel_features_flat.T) # (C, C)
        # Add a small epsilon to the diagonal for numerical stability (regularization)
        cov_matrix += torch.eye(cov_matrix.shape[0], device=device) * 1e-6
        inv_cov_matrix = torch.inverse(cov_matrix) # (C, C)
    except Exception as e:
        print(f"Error calculating inverse covariance matrix: {e}. Falling back to L2-like distance.")
        # Fallback if inverse fails (e.g., singular matrix even with regularization)
        mean_ref_feat = torch.mean(ref_pixel_features_flat, dim=0)
        return torch.norm(fobj_q_i_pixel_feat - mean_ref_feat, p=2) ** 2 # Squared L2 distance

    diff = fobj_q_i_pixel_feat - mean_ref_feat # (C,)
    # Mahalanobis distance squared: diff.T @ inv_cov @ diff
    mahalanobis_sq = torch.matmul(diff.unsqueeze(0), torch.matmul(inv_cov_matrix, diff.unsqueeze(1))).squeeze()

    return torch.sqrt(mahalanobis_sq) # Mahalanobis distance is sqrt of this

# --- Tu función build_reference_pixel_distributions (sin cambios en la función en sí) ---
def build_reference_pixel_distributions(all_fobj_r_list, target_mask_h, target_mask_w):
    """
    Builds pixel-wise mean and covariance for reference features across all reference images.
    Returns dictionaries of (mean, covariance) for each (h_feat, w_feat) position.
    """
    pixel_features_per_location = {} # Stores list of features for each (h,w) in feature map resolution

    # Initialize lists for each pixel location
    C = all_fobj_r_list[0].shape[1] if len(all_fobj_r_list) > 0 else 0
    if C == 0:
        print("Warning: No reference objects to build distributions from.")
        return {}
    
    for h_feat in range(target_mask_h):
        for w_feat in range(target_mask_w):
            pixel_features_per_location[(h_feat, w_feat)] = []

    # Collect features for each pixel location from all reference objects in all reference images
    for fobj_r_current_image in all_fobj_r_list: # (N_i, C, H_feat, W_feat)
        for n_idx in range(fobj_r_current_image.shape[0]): # Iterate through each reference object
            for h_feat in range(target_mask_h):
                for w_feat in range(target_mask_w):
                    pixel_feat = fobj_r_current_image[n_idx, :, h_feat, w_feat] # (C,)
                    pixel_features_per_location[(h_feat, w_feat)].append(pixel_feat)

    # Convert lists to tensors and calculate mean/covariance for each location
    ref_pixel_distributions = {}
    for (h_feat, w_feat), features_list in pixel_features_per_location.items():
        if len(features_list) > 0:
            features_tensor = torch.stack(features_list).to(device) # (Num_samples_at_pixel, C)
            ref_pixel_distributions[(h_feat, w_feat)] = features_tensor # Store the raw features for Mahalanobis
        else:
            ref_pixel_distributions[(h_feat, w_feat)] = None # No data for this pixel location

    return ref_pixel_distributions

# --- MODIFICADA: create_full_anomaly_map ---
# --- MODIFICADA: create_full_anomaly_map ---
def create_full_anomaly_map(M, masks_data, P_matrices, P_augmented_full_matrices,
                            fobj_q, all_fobj_r_list,
                            anomalous_ids, anomaly_detection_threshold,
                            image_original_shape, target_mask_h, target_mask_w,
                            ref_pixel_distributions, device,
                            global_anomaly_score_ceiling=None): # ¡NUEVO PARÁMETRO AQUÍ!
    """
    Creates the full anomaly map for the query image, and separate matched/unmatched maps.
    """
    # Initialize separate anomaly maps with zeros, same resolution as original image
    matched_anomaly_map = torch.zeros((image_original_shape[0], image_original_shape[1]), device=device, dtype=torch.float32)
    unmatched_anomaly_map = torch.zeros((image_original_shape[0], image_original_shape[1]), device=device, dtype=torch.float32)
    
    # Initialize a mask to keep track of processed pixels (optional, but good for debugging/completeness)
    processed_pixels_mask = torch.zeros((image_original_shape[0], image_original_shape[1]), dtype=torch.bool, device=device)

    # Iterar a través de cada objeto de consulta
    for q_idx in range(M):
        # Determine if the query object is matched or unmatched based on your existing logic
        is_matched = False
        best_overall_confidence = -1.0
        best_ref_match_idx = -1
        best_ref_image_idx = -1
        conf_to_trash_bin = -1.0

        for i, (P_current, P_augmented_current) in enumerate(zip(P_matrices, P_augmented_full_matrices)):
            if P_current.shape[0] == 0: continue # Skip empty P matrices

            M_current, N_current = P_current.shape
            if q_idx >= M_current: continue # Query object might not be in this P if M_current < M

            max_conf_to_real_ref_current, best_ref_obj_idx_current = P_current[q_idx, :].max(dim=0)
            conf_to_trash_bin_current = P_augmented_current[q_idx, N_current] # N_current is the trash bin column index

            if max_conf_to_real_ref_current > best_overall_confidence:
                best_overall_confidence = max_conf_to_real_ref_current
                best_ref_match_idx = best_ref_obj_idx_current
                best_ref_image_idx = i # Store which reference image this best match came from
            
            if conf_to_trash_bin_current > conf_to_trash_bin: # Keep track of the highest trash bin confidence
                conf_to_trash_bin = conf_to_trash_bin_current

        # Now, apply the anomaly condition for this specific query object
        if best_overall_confidence > anomaly_detection_threshold and \
           best_overall_confidence > conf_to_trash_bin:
            is_matched = True
            
        # Get the query object's original mask (full resolution)
        query_mask_dict = masks_data[q_idx]
        query_mask_binary = torch.from_numpy(query_mask_dict['segmentation']).to(device) # (Original_H, Original_W)

        # Get the query object's feature map (resized to TARGET_MASK_H, TARGET_MASK_W)
        query_obj_feat_map = fobj_q[q_idx] # (C, TARGET_MASK_H, TARGET_MASK_W)

        if is_matched and best_ref_image_idx != -1: # Matched object
            # Retrieve the matched reference object's feature map
            matched_ref_obj_feat_map = all_fobj_r_list[best_ref_image_idx][best_ref_match_idx] # (C, TARGET_MASK_H, TARGET_MASK_W)
            
            # Calculate L2 distance anomaly map for this matched object
            # This function will also resize it back to original image dimensions
            anomaly_score_map_obj_resized = calculate_anomaly_map_matched(
                query_obj_feat_map, matched_ref_obj_feat_map,
                image_original_shape, target_mask_h, target_mask_w # These target_mask_h/w are 128x128
            )
            
            # Apply this score map only within the query object's mask
            matched_anomaly_map[query_mask_binary] = anomaly_score_map_obj_resized[query_mask_binary]
            processed_pixels_mask[query_mask_binary] = True # Mark pixels as processed

        else: # Unmatched object (including those flagged as anomalous by previous logic)
            anomaly_score_map_obj_unmatched_low_res = torch.zeros((target_mask_h, target_mask_w), device=device, dtype=torch.float32)

            for h_feat in range(target_mask_h):
                for w_feat in range(target_mask_w):
                    q_pixel_feat = query_obj_feat_map[:, h_feat, w_feat] # (C,)
                    
                    if ref_pixel_distributions.get((h_feat, w_feat)) is not None:
                        ref_pixel_feats_flat = ref_pixel_distributions[(h_feat, w_feat)]
                        
                        # Only calculate Mahalanobis if enough samples, otherwise L2
                        if ref_pixel_feats_flat.shape[0] > ref_pixel_feats_flat.shape[1]: # Num samples > C (feature dim)
                            score = calculate_anomaly_map_unmatched_mahalanobis(q_pixel_feat, ref_pixel_feats_flat)
                        else: # Not enough samples for robust Mahalanobis, fall back to L2 to mean
                            mean_ref_feat = torch.mean(ref_pixel_feats_flat, dim=0)
                            score = torch.norm(q_pixel_feat - mean_ref_feat, p=2)
                    else: # No reference data for this pixel location, assign high anomaly score
                        score = torch.tensor(float('inf'), device=device)
                    
                    anomaly_score_map_obj_unmatched_low_res[h_feat, w_feat] = score
            
            # Resize this anomaly map to original image dimensions
            anomaly_score_map_obj_unmatched_resized = F.interpolate(
                anomaly_score_map_obj_unmatched_low_res.unsqueeze(0).unsqueeze(0),
                size=(image_original_shape[0], image_original_shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

            # Apply this score map only within the query object's mask
            unmatched_anomaly_map[query_mask_binary] = anomaly_score_map_obj_unmatched_resized[query_mask_binary]
            processed_pixels_mask[query_mask_binary] = True # Mark pixels as processed

    # Handle pixels not covered by any mask (if any remain, they should be considered normal or background)
    # For now, if a pixel isn't covered by any object mask, it remains 0 (initialized).

    # --- INICIO: SECCIÓN DE IMPRESIÓN DE SCORES CRUDOS ---
    # Calcular los máximos actuales para los mapas ANTES de cualquier clamping o normalización
    max_score_matched_raw = matched_anomaly_map[matched_anomaly_map != float('inf')].max().item() if matched_anomaly_map[matched_anomaly_map != float('inf')].numel() > 0 else 0.0
    max_score_unmatched_raw = unmatched_anomaly_map[unmatched_anomaly_map != float('inf')].max().item() if unmatched_anomaly_map[unmatched_anomaly_map != float('inf')].numel() > 0 else 0.0
    
    print(f"\n--- VALORES DE ANOMALÍA MÁXIMOS CRUDOS (ANTES DE CLAMPING/NORMALIZACIÓN) ---")
    print(f"Max score matched (raw): {max_score_matched_raw:.4f}")
    print(f"Max score unmatched (raw): {max_score_unmatched_raw:.4f}")
    print(f"------------------------------------------------------------------\n")
    # --- FIN: SECCIÓN DE IMPRESIÓN DE SCORES CRUDOS ---

    # --- NORMALIZACIÓN Y CLAMPING PARA CADA MAPA (Lógica modificada para usar global_anomaly_score_ceiling) ---
    
    # 1. Determinar el valor de referencia para el clamping y la normalización.
    if global_anomaly_score_ceiling is not None:
        effective_clamp_and_norm_value = global_anomaly_score_ceiling
    else:
        # Lógica adaptativa anterior (si no se especifica un valor global)
        max_score_matched_for_adaptive = matched_anomaly_map[matched_anomaly_map != float('inf')].max() if matched_anomaly_map[matched_anomaly_map != float('inf')].numel() > 0 else 0.0
        max_score_unmatched_for_adaptive = unmatched_anomaly_map[unmatched_anomaly_map != float('inf')].max() if unmatched_anomaly_map[unmatched_anomaly_map != float('inf')].numel() > 0 else 0.0
        
        effective_clamp_and_norm_value = max(max_score_matched_for_adaptive, max_score_unmatched_for_adaptive) * 1.5
    
    # Asegurarse de que el valor no sea demasiado pequeño para evitar división por cero
    if effective_clamp_and_norm_value < 1e-6:
        effective_clamp_and_norm_value = 1.0

    # 2. Aplicar el clamping (recortar los valores que exceden el límite superior)
    matched_anomaly_map_clamped = torch.clamp(matched_anomaly_map, min=0.0, max=effective_clamp_and_norm_value)
    unmatched_anomaly_map_clamped = torch.clamp(unmatched_anomaly_map, min=0.0, max=effective_clamp_and_norm_value)

    # 3. Normalizar los mapas resultantes a [0, 1]
    if effective_clamp_and_norm_value > 0:
        matched_anomaly_map_normalized = matched_anomaly_map_clamped / effective_clamp_and_norm_value
        unmatched_anomaly_map_normalized = unmatched_anomaly_map_clamped / effective_clamp_and_norm_value
    else:
        matched_anomaly_map_normalized = matched_anomaly_map_clamped
        unmatched_anomaly_map_normalized = unmatched_anomaly_map_clamped

    # Crear el mapa completo como la suma de los mapas normalizados
    full_anomaly_map = matched_anomaly_map_normalized + unmatched_anomaly_map_normalized
    # Recortar a 1.0 si la suma excede 1.0
    full_anomaly_map = torch.clamp(full_anomaly_map, min=0.0, max=1.0)
    
    # Retornar los tres mapas
    return matched_anomaly_map_normalized.cpu().numpy(), unmatched_anomaly_map_normalized.cpu().numpy(), full_anomaly_map.cpu().numpy()



# --- La función plot_anomaly_map (sin cambios en la función en sí) ---
def plot_anomaly_map(anomaly_map_np, image_np, save_path=None, title="Mapa de Anomalías"):
    """
    Plots the anomaly map overlayed on the original image.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    # Use a colormap that highlights high values (anomalies)
    plt.imshow(anomaly_map_np, cmap='jet', alpha=0.6)
    plt.colorbar(label='Puntuación de Anomalía')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Mapa de anomalías guardado en: {save_path}")
    plt.show()
    plt.close()


# --- Main AMM execution ---
# 1. Pre-calculate reference pixel distributions for unmatched object Mahalanobis distance
# Ya no es necesario pasar 'device' a build_reference_pixel_distributions si ya está definido globalmente
ref_pixel_distributions = build_reference_pixel_distributions(all_fobj_r_list, TARGET_MASK_H, TARGET_MASK_W)

# 1.2. Create the full anomaly map, and now also separate matched/unmatched maps
GLOBAL_ANOMALY_SCORE_CEILING = 30 # <--- ¡AJUSTA ESTE VALOR! Prueba con 20.0, 10.0, etc. eleva el valor de las confianzas e inmoviliza a trash
matched_anomaly_map, unmatched_anomaly_map, full_query_anomaly_map = create_full_anomaly_map(
    M, masks_data, P_matrices, P_augmented_full_matrices,
    fobj_q, all_fobj_r_list,
    anomalous_ids, anomaly_detection_threshold,
    image_for_sam_np.shape, TARGET_MASK_H, TARGET_MASK_W,
    ref_pixel_distributions, device,
    global_anomaly_score_ceiling=GLOBAL_ANOMALY_SCORE_CEILING # ¡PASAMOS EL NUEVO PARÁMETRO!
)

# 2. Create the full anomaly map, and now also separate matched/unmatched maps
matched_anomaly_map, unmatched_anomaly_map, full_query_anomaly_map = create_full_anomaly_map(
    M, masks_data, P_matrices, P_augmented_full_matrices,
    fobj_q, all_fobj_r_list,
    anomalous_ids, anomaly_detection_threshold, # Use your existing threshold here
    image_for_sam_np.shape, TARGET_MASK_H, TARGET_MASK_W,
    ref_pixel_distributions, device
)

# 3. Plot the final anomaly map (General)
output_full_anomaly_map_filename = os.path.join(plot_save_directory_on_server, 'final_anomaly_map_full_cut_000.png')
plot_anomaly_map(full_query_anomaly_map, image_for_sam_np, save_path=output_full_anomaly_map_filename, title="Mapa de Anomalías General")

# 4. Plot the Matched Anomaly Map
output_matched_anomaly_map_filename = os.path.join(plot_save_directory_on_server, 'final_anomaly_map_matched_cut_000.png')
plot_anomaly_map(matched_anomaly_map, image_for_sam_np, save_path=output_matched_anomaly_map_filename, title="Mapa de Anomalías (Objetos Emparejados)")

# 5. Plot the Unmatched Anomaly Map
output_unmatched_anomaly_map_filename = os.path.join(plot_save_directory_on_server, 'final_anomaly_map_unmatched_cut_000.png')
plot_anomaly_map(unmatched_anomaly_map, image_for_sam_np, save_path=output_unmatched_anomaly_map_filename, title="Mapa de Anomalías (Objetos No Emparejados)")

print("--- Módulo de Medición de Anomalías (AMM) Completado ---")



end_time = time.time()
total_execution_time = (end_time - start_time) + time_knn_dist + (end_time_sam - start_time_sam)
print(f"\nTiempo total de ejecución del script: {total_execution_time:.4f} segundos")

print(f"Finalizado")



