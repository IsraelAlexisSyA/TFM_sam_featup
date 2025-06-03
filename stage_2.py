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
end_time = time.time()
plot_feats(unnorm(input_query_tensor)[0], query_lr_feats_featup[0], query_hr_feats[0])
# Guardar el plot en un archivo
output_query_plot_filename = os.path.join(plot_save_directory_on_server, 'query_image_features_plot.png')
plt.tight_layout()
plt.savefig(output_query_plot_filename)
print(f"Plot de características de la imagen de consulta guardado en: {output_query_plot_filename}")
plt.close()


# 6.2 A similares imagenes

print("Imágenes similares:", imagenes_similares)
for i, similar_image_path in enumerate(rutas_imagenes_similares):  # Usa los paths
    # Cargar y transformar la imagen similar
    input_similar_tensor = transform(Image.open(similar_image_path).convert("RGB")).unsqueeze(0).to(device)
    
    # Aplicar FeatUp para obtener características de baja y alta resolución
    similar_lr_feats, similar_hr_feats = apply_featup_hr(similar_image_path, upsampler, transform, device)
    
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
import matplotlib
matplotlib.use('Agg')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Funciones de visualización (las mismas que en tu código original, solo para referencia)
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def save_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, output_filename="mask_output.png"):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        else:
            plt.title(f"Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        
        # Save the plot to a .png file
        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"Mask plot saved to: {output_filename}")
        plt.close()

checkpoint = "/home/imercatoma/sam2_repo_independent/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

try:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 cargado correctamente con SAM2ImagePredictor")
except FileNotFoundError as e:
    print(f"Error: No se encontraron los archivos del modelo SAM2: {e}")
    exit()
except Exception as e:
    print("Error al cargar SAM2:")
    print(e)
    exit()

# Cargar la imagen de consulta para SAM
# Asegúrate de que 'query_image_path' esté definido y sea accesible
try:
    image_for_sam = Image.open(query_image_path).convert("RGB")
    image_for_sam_np = np.array(image_for_sam)
except FileNotFoundError:
    print(f"Error: No se encontró la imagen en la ruta: {query_image_path}")
    exit()

try:
    predictor.set_image(image_for_sam_np) # Procesa la imagen para obtener los embeddings
except Exception as e:
    print("Error al procesar la imagen para SAM:")
    print(e)
    exit()

# Definir los puntos de entrada para SAM
# Aquí es donde especificas dos puntos: uno para el objeto y otro para el fondo
# Ajusta estas coordenadas según el objeto que quieras segmentar en tu imagen de consulta
# Coordenada del punto para el objeto (etiqueta 1)
point_coords = np.array([[500, 500], [600, 600]]) # Ejemplo: [y, x] para dos puntos
# Etiquetas de los puntos: 1 para punto positivo (dentro del objeto), 0 para punto negativo (fuera del objeto)
point_labels = np.array([1, 1]) # El primer punto es positivo, el segundo es negativo

# Generar la máscara utilizando los puntos
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=False, # Genera una sola máscara de salida
)

# Mostrar la máscara resultante
# Utilizamos la función show_masks adaptada para mostrar solo los puntos
output_mask_filename = os.path.join(plot_save_directory_on_server, 'sam_mask_output.png')
save_masks(image_for_sam_np, masks, scores, point_coords=point_coords, input_labels=point_labels, output_filename=output_mask_filename)

end_time_sam = time.time()
print(f"Tiempo de ejecución de SAM: {end_time_sam - start_time_sam:.4f} segundos")

total_execution_time = (end_time - start_time) + time_knn_dist + (end_time_sam - start_time_sam)
print(f"\nTiempo total de ejecución del script: {total_execution_time:.4f} segundos")



end_time_sam = time.time()






#end_time = time.time() 

total_execution_time = (end_time - start_time)+(time_knn_dist)
print(f"\nTiempo total de ejecución del script: {total_execution_time:.4f} segundos")
print(f"Finalizado SAm")