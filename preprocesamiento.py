# Importamos PyTorch, que usaremos para el modelo y el procesamiento de tensores
import torch

# Importamos las transformaciones estándar de imágenes de torchvision
import torchvision.transforms as transforms

# Importamos los modelos preentrenados de torchvision
from torchvision import models

# PIL se usa para abrir y procesar imágenes
from PIL import Image

# Función para calcular la distancia coseno entre vectores
from scipy.spatial.distance import cosine

# ========================================
# 1. Definimos una transformación para las imágenes
# ========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionamos la imagen a 224x224 píxeles (tamaño que espera ResNet)
    transforms.ToTensor(),          # Convertimos la imagen PIL a un tensor PyTorch (valores de 0 a 1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Normalizamos usando la media
                         std=[0.229, 0.224, 0.225])    # y desviación estándar del dataset ImageNet
])

# ========================================
# 2. Función para obtener el embedding (vector de características) de una imagen
# ========================================

def get_image_embedding(img_path, model):
    img = Image.open(img_path).convert('RGB')   # Abrimos la imagen y la convertimos a RGB
    img_t = transform(img).unsqueeze(0)         # Aplicamos la transformación y agregamos una dimensión batch (1, C, H, W)
    with torch.no_grad():                       # Desactivamos el cálculo de gradientes (más eficiente)
        embedding = model(img_t).squeeze().numpy()  # Pasamos la imagen por el modelo y convertimos el tensor a array NumPy
    return embedding                            # Devolvemos el vector de características (embedding)

# ========================================
# 3. Cargamos el modelo ResNet50 preentrenado en ImageNet y lo preparamos
# ========================================

resnet = models.resnet50(pretrained=True)       # Cargamos el modelo ResNet50 con pesos preentrenados
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Eliminamos la capa final (clasificación) para obtener solo características
resnet.eval()                                   # Ponemos el modelo en modo evaluación (no entrenamiento)

# ========================================
# 4. Obtenemos los vectores de características de dos imágenes
# ========================================

embedding1 = get_image_embedding('Image\\etiqueta_base.jpg', resnet)  # Embedding de la primera imagen
embedding2 = get_image_embedding('Image\\etiqueta_foto.jpg', resnet)  # Embedding de la segunda imagen

# ========================================
# 5. Calculamos la similitud entre los embeddings usando la distancia coseno
# ========================================

similaridad = 1 - cosine(embedding1, embedding2)  # La similitud coseno es 1 - distancia coseno
print(f"Similitud entre imágenes: {similaridad:.4f}")  # Mostramos la similitud con 4 decimales

# ========================================
# 6. Evaluamos si las imágenes son similares según un umbral
# ========================================

if similaridad > 0.85:                             # Si la similitud es mayor a 0.85, consideramos las imágenes similares
    print("✅ Las imágenes son visualmente del mismo tipo.")
else:
    print("❌ Las imágenes parecen ser diferentes.")
