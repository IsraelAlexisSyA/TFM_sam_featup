import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

from featup.util import norm, unnorm

from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from scipy.stats import median_abs_deviation

# --- Configuración ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224 # Tamaño de entrada para DINOv2
BACKBONE_PATCH_SIZE = 14 # Tamaño de parche para DINOv2 ViT-S/14
use_norm = True

# Dimensiones espaciales de los mapas de características de baja resolución (H', W')
H_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 14 = 16
W_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 16 = 16

# Directorios
TRAIN_GOOD_DIR = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/train/good'
TEST_CRACK_DIR = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/test/cut' # Tu directorio objetivo
PLOT_SAVE_ROOT_DIR = '/home/imercatoma/FeatUp/plots_anomaly_distances' # Carpeta raíz para todos los plots
os.makedirs(PLOT_SAVE_ROOT_DIR, exist_ok=True)
print(f"Directorio raíz de guardado de plots creado/verificado: '{PLOT_SAVE_ROOT_DIR}'")

# Rutas de archivo para cargar los mapas de características del Coreset
core_bank_filenames_file = os.path.join(TRAIN_GOOD_DIR, 'core_bank_filenames.pt')
coreset_relevant_flat_features_bank_file = os.path.join(TRAIN_GOOD_DIR, 'coreset_relevant_flat_features_bank.pt')
template_features_bank_coreset_file = os.path.join(TRAIN_GOOD_DIR, 'template_features_bank_coreset.pt') # Este es tu M

# --- Cargar Datos del Coreset (Matriz 'M' para KNN) ---
print("Cargando datos del coreset relevante y banco de características (M)...")
coreset_relevant_filenames = []
coreset_relevant_flat_features_bank = None
coreset_features = None # Esta será la matriz 'M' para KNN

try:
    coreset_relevant_filenames = torch.load(core_bank_filenames_file)
    coreset_relevant_flat_features_bank = torch.load(coreset_relevant_flat_features_bank_file).to(device)
    coreset_features = torch.load(template_features_bank_coreset_file).to(device) # Tu 'M'
    
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
print(f"Características del coreset movidas a CPU. Forma: {coreset_features_cpu.shape}")

# Inicializar NearestNeighbors finder una sola vez
nn_finder = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(coreset_features_cpu)
print("NearestNeighbors finder inicializado con características del coreset.")

# --- Cargar Modelo DINOv2 ---
print("Cargando modelo DINOv2 para extracción de características...")
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
dinov2_model = upsampler.model # Obtener el modelo base DINOv2 del upsampler
dinov2_model.eval() # Poner el modelo en modo de evaluación
print("Modelo DINOv2 cargado.")

# --- Transformación de Imagen ---
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(), # Escala píxeles a [0, 1] y cambia a (C, H, W)
    norm # Aplica normalización de ImageNet (media/std)
])

# --- Función Principal para Obtener Puntuaciones de Anomalía para una Imagen ---
def get_anomaly_scores_for_image(image_path, model, image_transform, nn_finder_instance, H_prime, W_prime, device):
    """
    Extrae características DINOv2 para una sola imagen, calcula puntuaciones de anomalía por parche
    basadas en la distancia al coreset.
    Devuelve tanto patch_anomaly_scores (sin ordenar) como sorted_patch_anomaly_scores.
    """
    # 1. Extraer características DINOv2 para la imagen de consulta
    try:
        input_tensor = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"  Error cargando/transformando imagen {os.path.basename(image_path)}: {e}")
        return None, None

    with torch.no_grad():
        features_lr = model(input_tensor) # (1, C, H', W')

    # 2. Aplanar parches
    query_patches_flat = features_lr.squeeze(0).permute(1, 2, 0).reshape(-1, features_lr.shape[1])
    query_patches_flat_cpu = query_patches_flat.cpu().numpy()

    # 3. Calcular distancias a los vecinos más cercanos en el coreset
    distances_to_nn, _ = nn_finder_instance.kneighbors(query_patches_flat_cpu)
    patch_anomaly_scores = distances_to_nn.flatten() # Forma (H'*W',)
    # NO TOCAR: patch_anomaly_scores ya está en el orden espacial correcto.

    # 4. Ordenar las puntuaciones de anomalía en orden descendente (para Q-score y otras métricas que lo necesiten)
    sorted_patch_anomaly_scores = np.sort(patch_anomaly_scores)[::-1]

    return patch_anomaly_scores, sorted_patch_anomaly_scores # Devolver ambos

# --- Funciones de Cálculo de Métricas (de pasos anteriores) ---
def calculate_rms(data):
    """Calcula el Root Mean Square (RMS) de un array de datos."""
    return np.sqrt(np.mean(data**2))

def calculate_mad(data):
    """Calcula la Desviación Absoluta Mediana (MAD) de un array de datos."""
    return median_abs_deviation(data)

def calculate_median(data):
    """Calcula la Mediana de un array de datos."""
    return np.median(data)

def calculate_quartile(data, q=25):
    """Calcula un cuartil específico (ej., 25º percentil para Q1)."""
    return np.percentile(data, q)

# --- Funciones para los filtros de concentración/extensión y magnitud ---
def calculate_spatial_variance_of_top_patches(patch_anomaly_scores, top_percentage=5.5):
    """
    Calcula la varianza espacial de un porcentaje de los parches con mayor anomalía.
    patch_anomaly_scores: Array 1D de puntuaciones de anomalía por parche (sin ordenar, en orden espacial).
    top_percentage: Porcentaje de los parches más anómalos a considerar (ej., 5.5 para el 5.5%).
    Devuelve la suma de las desviaciones estándar de las coordenadas X e Y.
    """
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0:
        return np.nan

    num_patches = patch_anomaly_scores.size
    
    if num_patches != H_prime * W_prime:
        print(f"  Advertencia: Número de parches ({num_patches}) no coincide con H'*W' ({H_prime*W_prime}).")

    num_top = max(1, int(num_patches * top_percentage / 100)) # Al menos 1 parche

    top_patch_indices = np.argsort(patch_anomaly_scores)[-num_top:]

    row_coords = top_patch_indices // W_prime
    col_coords = top_patch_indices % W_prime

    std_rows = np.std(row_coords) if len(row_coords) > 1 else 0.0
    std_cols = np.std(col_coords) if len(col_coords) > 1 else 0.0

    return std_rows + std_cols

def calculate_active_patches_count_relative_threshold(patch_anomaly_scores, relative_threshold_percentage):
    """
    Calcula el número de parches cuyas puntuaciones de anomalía están por encima de un
    porcentaje del valor máximo de puntuación de anomalía en la imagen.
    """
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0:
        return 0

    max_val_in_image = np.max(patch_anomaly_scores)
    
    if max_val_in_image == 0:
        return 0
        
    threshold_val = max_val_in_image * relative_threshold_percentage
    
    active_patches = patch_anomaly_scores[patch_anomaly_scores > threshold_val]
    return len(active_patches)

def calculate_top_percent_average_anomaly(patch_anomaly_scores, top_percent=1):
    """
    Calcula el promedio de las puntuaciones de anomalía del porcentaje superior de parches.
    patch_anomaly_scores: Array 1D de puntuaciones de anomalía por parche (sin ordenar).
    top_percent: Porcentaje de los parches con mayor anomalía a promediar (ej., 1 para el 1%).
    """
    if patch_anomaly_scores is None or patch_anomaly_scores.size == 0:
        return 0.0
    
    num_patches = patch_anomaly_scores.size
    num_top = max(1, int(num_patches * top_percent / 100)) # Asegurarse de que al menos 1 parche

    # Obtener los top N valores y calcular su promedio
    sorted_scores = np.sort(patch_anomaly_scores)[::-1] # Ordenar de mayor a menor
    top_n_scores = sorted_scores[:num_top]
    
    return np.mean(top_n_scores)


# --- BARRIDO POR LOTES Y RECOLECCIÓN DE DATOS ---
image_names_processed = []
rms_mad_distances = []
rms_median_distances = []
rms_q1_distances = []
spatial_variances = []
active_patches_counts = []
top_1_percent_averages = []
image_classifications = [] # Nueva lista para almacenar la clasificación (0=Buena, 1=Anómala)

# --- UMBRAL RELATIVO PARA PARCHES ACTIVOS ---
RELATIVE_ACTIVE_PATCH_THRESHOLD_PERCENTAGE = 0.80

print(f"\nIniciando procesamiento por lotes de imágenes desde: {TEST_CRACK_DIR}")
try:
    test_image_files = [f for f in os.listdir(TEST_CRACK_DIR) if f.lower().endswith('.png')]
    test_image_files.sort() # Asegurar un orden consistente
    if not test_image_files:
        print(f"No se encontraron imágenes .png en: {TEST_CRACK_DIR}")
except FileNotFoundError:
    print(f"Error: Directorio '{TEST_CRACK_DIR}' no encontrado.")
    test_image_files = []

for img_file in test_image_files:
    full_image_path = os.path.join(TEST_CRACK_DIR, img_file)
    print(f"\n--- Procesando imagen: {img_file} ---")

    current_patch_anomaly_scores, current_sorted_patch_anomaly_scores = get_anomaly_scores_for_image(
        full_image_path, dinov2_model, transform, nn_finder, H_prime, W_prime, device
    )

    if current_patch_anomaly_scores is None:
        print(f"  Saltando {img_file} debido a un error de procesamiento.")
        continue # Saltar a la siguiente imagen si hubo un error

    # --- Normalizar Puntuaciones de Anomalía (para métricas de distancia RMS/MAD/Mediana/Q1) ---
    min_val = np.min(current_sorted_patch_anomaly_scores)
    max_val = np.max(current_sorted_patch_anomaly_scores)

    if max_val == min_val:
        normalized_data = np.zeros_like(current_sorted_patch_anomaly_scores, dtype=float)
    else:
        normalized_data = (current_sorted_patch_anomaly_scores - min_val) / (max_val - min_val)

    # --- Calcular Métricas de Distancia (ya implementadas) ---
    A_rms = calculate_rms(normalized_data)
    B_mad = calculate_mad(normalized_data)
    C_median = calculate_median(normalized_data)
    D_q1_normalized = calculate_quartile(normalized_data, q=25) 

    dist_rms_mad = A_rms - B_mad
    dist_rms_median = A_rms - C_median
    dist_rms_q1 = A_rms - D_q1_normalized

    # --- Calcular Métricas de Concentración/Extensión y Magnitud Absoluta ---
    spatial_var = calculate_spatial_variance_of_top_patches(current_patch_anomaly_scores) 
    
    active_count = calculate_active_patches_count_relative_threshold(
        current_patch_anomaly_scores, relative_threshold_percentage=RELATIVE_ACTIVE_PATCH_THRESHOLD_PERCENTAGE
    ) 
    
    top_1_avg = calculate_top_percent_average_anomaly(current_patch_anomaly_scores, top_percent=1) # Promedio del top 1% de anomalía

    # Almacenar resultados (antes de la clasificación para tener todos los datos)
    image_names_processed.append(img_file)
    rms_mad_distances.append(dist_rms_mad)
    rms_median_distances.append(dist_rms_median)
    rms_q1_distances.append(dist_rms_q1)
    spatial_variances.append(spatial_var)
    active_patches_counts.append(active_count)
    top_1_percent_averages.append(top_1_avg)

    print(f"  Métricas calculadas para {img_file}:")
    print(f"    Promedio Top 1% Anomalía (para clasificación): {top_1_avg:.4f}")
    print(f"    Active Patches Count (> {RELATIVE_ACTIVE_PATCH_THRESHOLD_PERCENTAGE*100:.0f}% del Max): {active_count}")
    print(f"    Varianza Espacial (Top 5.5%): {spatial_var:.4f}")
    print(f"    Distancia (RMS - Mediana): {dist_rms_median:.4f}")
    print(f"    Distancia (RMS - MAD): {dist_rms_mad:.4f}")

    # --- Lógica de Clasificación ---
    classification = 0 # Asumir "Buena" (0) por defecto

    # 1. Si el top maxscore q1% (que es top_1_avg) es >= 0.30 -> Anomalía Grande
    if top_1_avg >= 0.30:
        classification = 1 # Anómala
        print(f"  Clasificación: ANOMALÍA GRANDE (Top 1% Avg: {top_1_avg:.4f} >= 0.30)")
    
    # 2. Si top maxscore q1% (top_1_avg) es < 0.30 y >= 0.17
    elif 0.17 <= top_1_avg < 0.30:
        print(f"  Clasificación: Entrando en evaluación de anomalía leve/buena (Top 1% Avg: {top_1_avg:.4f})")
        
        # a. Si active_patches_count_per_test_image > 5 -> Anomalía Leve
        if active_count > 5: # Changed from > 4 to > 5
            classification = 1 # Anómala
            print(f"    -> ANOMALÍA LEVE (Parches Activos: {active_count} > 5)")
        else: # Si no cumple la condición a. (active_count <= 5)
            print(f"    -> Parches Activos ({active_count}) <= 5. Evaluando condiciones de 'buena'.")
            
# II. Si la diferencia entre el RMS y la mediana es igual o menor a 0.055 es buena (Principal)
            if dist_rms_median <= 0.055:
                print(f"      - Condición Buena II (RMS - Mediana <= 0.055): True ({dist_rms_median:.4f})")
                
                # Si es TRUE, entonces se evalúan las siguientes dos condiciones
                cond_I_met = spatial_var >= 5.5
                print(f"      - Condición Buena I (Varianza Espacial >= 5.5): {'True' if cond_I_met else 'False'} ({spatial_var:.2f})")
                
                cond_III_met = dist_rms_mad >= 0.21
                print(f"      - Condición Buena III (RMS - MAD >= 0.21): {'True' if cond_III_met else 'False'} ({dist_rms_mad:.4f})")
                
                # CAMBIO AQUÍ: Al menos una (I o III) debe ser verdadera para que sea BUENA
                if cond_I_met or cond_III_met: # Changed 'and' to 'or'
                    classification = 0 # Buena
                    print(f"    -> IMAGEN BUENA (Condición II True, y al menos una de I o III es True)")
                else:
                    classification = 1 # Anomalía Leve
                    print(f"    -> ANOMALÍA LEVE (Condición II True, pero ni I ni III son True)")
            else: # Si Condición II (RMS - Mediana <= 0.055) es False
                classification = 1 # Anomalía Leve automáticamente
                print(f"      - Condición Buena II (RMS - Mediana <= 0.055): False ({dist_rms_median:.4f})")
                print(f"    -> ANOMALÍA LEVE (Condición II es False, clasificación automática como anomalía leve)")
                
    # 3. Si el q 1% (top_1_avg) es menor a 0.17 es considerado buena automaticamente
    elif top_1_avg < 0.17:
        classification = 0 # Buena
        print(f"  Clasificación: BUENA (Top 1% Avg: {top_1_avg:.4f} < 0.17)")
    
    image_classifications.append(classification)

# --- Graficar Resultados ---

if not image_names_processed:
    print("\nNo se procesaron datos de imagen con éxito para graficar.")
else:
    num_images = len(image_names_processed)
    fig_width = max(12, num_images * 0.8) # Ancho dinámico para plots
    x_positions = np.arange(num_images)

    # --- Plot 1: Distancias RMS - MAD (Existente) ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, rms_mad_distances, color='purple', width=0.6)
    plt.ylabel('Distancia (RMS - MAD)', fontsize=12)
    plt.title('Distancia (RMS - MAD) por Imagen', fontsize=14)
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
    print(f"\nPlot 'RMS - MAD por Imagen' guardado en: '{output_path_rms_mad}'")
    plt.close()

    # --- Plot 2: Distancias RMS - Mediana (Existente) ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, rms_median_distances, color='orange', width=0.6)
    plt.ylabel('Distancia (RMS - Mediana)', fontsize=12)
    plt.title('Distancia (RMS - Mediana) por Imagen', fontsize=14)
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
    print(f"Plot 'RMS - Mediana por Imagen' guardado en: '{output_path_rms_median}'")
    plt.close()

    # --- Plot 3: Distancias RMS - 1er Cuartil (Q1) (Existente) ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, rms_q1_distances, color='cyan', width=0.6)
    plt.ylabel('Distancia (RMS - 1er Cuartil)', fontsize=12)
    plt.title('Distancia (RMS - 1er Cuartil) por Imagen', fontsize=14)
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
    print(f"Plot 'RMS - 1er Cuartil por Imagen' guardado en: '{output_path_rms_q1}'")
    plt.close()

    # --- Plot 4: Varianza Espacial de Parches Más Anómalos ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, spatial_variances, color='salmon', width=0.6)
    plt.ylabel('Varianza Espacial (suma std_x + std_y)', fontsize=12)
    plt.title('Varianza Espacial de Top 5.5% Parches Anómalos por Imagen', fontsize=14)
    plt.xticks(x_positions, image_names_processed, rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    output_path_spatial_variance = os.path.join(PLOT_SAVE_ROOT_DIR, 'spatial_variance_top_patches_per_test_image.png')
    plt.savefig(output_path_spatial_variance)
    print(f"Nuevo Plot 'Varianza Espacial' guardado en: '{output_path_spatial_variance}'")
    plt.close()

    # --- Plot 5: Conteo de Parches Activos (con Umbral Relativo al Máximo) ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, active_patches_counts, color='forestgreen', width=0.6)
    plt.ylabel(f'Parches Activos (> {RELATIVE_ACTIVE_PATCH_THRESHOLD_PERCENTAGE*100:.0f}% del Max)', fontsize=12)
    plt.title('Número de Parches Activos por Imagen', fontsize=14)
    plt.xticks(x_positions, image_names_processed, rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    output_path_active_patches = os.path.join(PLOT_SAVE_ROOT_DIR, 'active_patches_count_per_test_image.png')
    plt.savefig(output_path_active_patches)
    print(f"Nuevo Plot 'Conteo de Parches Activos' guardado en: '{output_path_active_patches}'")
    plt.close()

    # --- Plot 6: Promedio Top 1% de Anomalía ---
    plt.figure(figsize=(fig_width, 7))
    bars = plt.bar(x_positions, top_1_percent_averages, color='dodgerblue', width=0.6)
    plt.ylabel('Promedio Puntuación Anomalía (Top 1%)', fontsize=12)
    plt.title('Promedio Puntuación Anomalía del Top 1% de Parches por Imagen', fontsize=14)
    plt.xticks(x_positions, image_names_processed, rotation=60, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    output_path_top_1_avg = os.path.join(PLOT_SAVE_ROOT_DIR, 'top_1_percent_average_anomaly_per_test_image.png')
    plt.savefig(output_path_top_1_avg)
    print(f"Nuevo Plot 'Promedio Top 1% Anomalía' guardado en: '{output_path_top_1_avg}'")
    plt.close()

    # --- NUEVO Plot 7: Clasificación de Anomalías (0=Buena, 1=Anómala) ---
    plt.figure(figsize=(fig_width, 7))
    colors = ['green' if c == 0 else 'red' for c in image_classifications] # Green for Good (0), Red for Anomalous (1)
    bars = plt.bar(x_positions, image_classifications, color=colors, width=0.6)
    plt.ylabel('Clasificación (0: Buena, 1: Anómala)', fontsize=12)
    plt.title('Clasificación de Anomalías por Imagen', fontsize=14)
    plt.xticks(x_positions, image_names_processed, rotation=60, ha='right', fontsize=10)
    plt.yticks([0, 1], ['Buena', 'Anómala'], fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    output_path_classification = os.path.join(PLOT_SAVE_ROOT_DIR, 'anomaly_classification_per_test_image.png')
    plt.savefig(output_path_classification)
    print(f"Nuevo Plot 'Clasificación de Anomalías' guardado en: '{output_path_classification}'")
    plt.close()

print("\nAnálisis de detección de anomalías por lotes y métricas completado.")