import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.metrics.pairwise import euclidean_distances
import time
#start_time = time.time() 
# Importar las utilidades de FeatUp para normalización/desnormalización
from featup.util import norm, unnorm
from featup.plotting import plot_feats # Asegúrate de que esta importación sea correcta

# --- Configuración Inicial ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224 # Tamaño de entrada para DINOv2
BACKBONE_PATCH_SIZE = 14 # Tamaño de parche para DINOv2 ViT-S/14
use_norm = True # Coherente con tu enfoque

# Las dimensiones espaciales de los mapas de características de baja resolución (H', W')
H_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 14 = 16
W_prime = input_size // BACKBONE_PATCH_SIZE # 224 // 14 = 16

# Directorio de imágenes (ajusta según tu estructura)
directorio_imagenes = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/train/good'

# Rutas de archivo para cargar los mapas de características completos relevantes para el Coreset
core_bank_features_file = os.path.join(directorio_imagenes, 'core_bank_features.pt')
core_bank_filenames_file = os.path.join(directorio_imagenes, 'core_bank_filenames.pt')
# --- NUEVO: Ruta para el banco de características del coreset relevante, ya aplanado y apilado ---
coreset_relevant_flat_features_bank_file = os.path.join(directorio_imagenes, 'coreset_relevant_flat_features_bank.pt')


# --- Cargar los datos del coreset relevante ---
print("Cargando datos del coreset relevante...")
coreset_relevant_filenames = []
coreset_relevant_flat_features_bank = None # Este será el tensor aplanado y apilado

try:
    coreset_relevant_filenames = torch.load(core_bank_filenames_file)
    # Cargar el banco de características ya aplanado y apilado
    coreset_relevant_flat_features_bank = torch.load(coreset_relevant_flat_features_bank_file).to(device)

    print(f"Banco de características relevante (aplanado y apilado) cargado. Dimensión: {coreset_relevant_flat_features_bank.shape}")
    print(f"Número de nombres de archivo relevantes cargados: {len(coreset_relevant_filenames)}")
    if coreset_relevant_filenames:
        print(f"Ejemplo de nombre de archivo relevante cargado: {coreset_relevant_filenames[0]}")

except FileNotFoundError as e:
    print(f"Error al cargar archivos del coreset relevante: {e}. Asegúrate de que Stage 1 se haya ejecutado correctamente y los archivos existan.")
    exit() # Salir si los archivos esenciales no se encuentran
except Exception as e:
    print(f"Ocurrió un error al cargar o procesar los archivos del coreset relevante: {e}")
    exit()


# Define donde quieres guardar el plot
plot_save_directory_on_server = '/home/imercatoma/FeatUp/plots_output'
output_plot_filename = os.path.join(plot_save_directory_on_server, 'query_image_000_plot.png')
os.makedirs(plot_save_directory_on_server, exist_ok=True)

# --- Extraer características de la imagen de consulta y buscar similares ---
query_image_path = '/home/imercatoma/FeatUp/datasets/mvtec_anomaly_detection/hazelnut/test/good/000.png'
query_img_pil = Image.open(query_image_path).convert('RGB')

# Mostrar y guardar la imagen de consulta
plt.imshow(query_img_pil)
plt.title('Imagen de Consulta')
plt.axis('off')
plt.savefig(output_plot_filename)
print(f"Plot de la imagen de consulta guardado en: {output_plot_filename}")
plt.close() # Cerrar el plot para liberar memoria

# --- Cargar modelo DINOv2 (a través de FeatUp) ---
print("Cargando modelo DINOv2 para extracción de características de consulta...")
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
dinov2_model = upsampler.model # Obtiene el modelo base DINOv2 del upsampler
dinov2_model.eval() # Pone el modelo en modo de evaluación
print("Modelo DINOv2 cargado.")

# --- Transformación Única para todas las imágenes ---
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(), # Escala píxeles a [0, 1] y cambia a (C, H, W)
    norm # Aplica normalización por media/std (normalización ImageNet)
])

def extract_dinov2_features_lr(image_path, model, image_transform, device):
    """Extrae características de baja resolución de DINOv2 usando la transformación dada."""
    input_tensor = image_transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    return features.cpu() # Mantener en CPU hasta que sea necesario

# Extraer características de la imagen de consulta
query_lr_features = extract_dinov2_features_lr(query_image_path, dinov2_model, transform, device)
print(f"Dimensiones de características de consulta (baja resolución): {query_lr_features.shape}")


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
    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False) # apply_postprocessing=False para AutomaticMaskGenerator
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

# --- Parámetros para el grid de puntos ---
# Puedes ajustar estos valores según la densidad de puntos que desees
# Se usará 'points_per_side' para definir una cuadrícula cuadrada.
# Por ejemplo, si pones 16, se generará una cuadrícula de 16x16 puntos.
points_grid_density = 16 # 16 - 12 etc Número de puntos a lo largo de un lado del grid

# Inicializar el generador de máscaras automático
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=points_grid_density, # Usamos la variable que definimos
    points_per_batch=256, # 64 - 256 Número de puntos procesados en cada lote (ajustable según tu GPU)
    pred_iou_thresh=0.88, # Umbral de confianza para filtrar máscaras
    stability_score_thresh=0.95, # Umbral de estabilidad para filtrar máscaras
    crop_n_layers=0, # Desactiva el cropping batch de recorte 
    #crop_n_points_downscale_factor (por defecto 1) depende de crop_n_layers > 1
    min_mask_region_area=25.0, # Área mínima de la máscara para filtrar (en píxeles)
)

# CORRECCIÓN AQUÍ: Usamos la variable 'points_grid_density' que contiene el valor
print(f"Generando máscaras con un grid de {points_grid_density}x{points_grid_density} puntos...")

# Generar máscaras
masks_data = mask_generator.generate(image_for_sam_np)

print(f"Se generaron {len(masks_data)} máscaras.")

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


# --- Aplicando SAM MASK a las imágenes similares ---
print("\nGenerando y visualizando máscaras SAM para las imágenes similares...")
similar_masks_raw_list = []
for i, similar_image_path in enumerate(rutas_imagenes_similares):
    try:
        img_similar_pil = Image.open(similar_image_path).convert('RGB')
        image_np_similar_for_sam = np.array(img_similar_pil)

        print(f"--- Procesando: {os.path.basename(similar_image_path)} ---") # Add a clear separator
        current_similar_masks_data = mask_generator.generate(image_np_similar_for_sam)
        similar_masks_raw_list.append(current_similar_masks_data) # Guardar las máscaras generadas para esta imagen
        print(f"Dimensiones imagen similar para SAM (np.array): {image_np_similar_for_sam.shape}")
        print(f"Se generaron {len(current_similar_masks_data)} máscaras para {os.path.basename(similar_image_path)}.")

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


end_time = time.time() # Dummy value
total_execution_time = (end_time - start_time) + time_knn_dist + (end_time_sam - start_time_sam)
print(f"\nTiempo total de ejecución del script: {total_execution_time:.4f} segundos")


print(f"Finalizado SAm")