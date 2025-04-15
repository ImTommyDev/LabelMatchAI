# ========================================
# Script principal para comparar dos etiquetas visualmente.
# Lo que hago aquí es: primero recortar la etiqueta de una imagen grande,
# luego comparar esa etiqueta recortada con una imagen buena (plantilla) usando un modelo de visión por computadora.
# ========================================

# ==== Importo las librerías necesarias ====

import torch  # Para trabajar con tensores y modelos preentrenados
import torchvision.transforms as transforms  # Para preparar las imágenes
from torchvision import models  # Para cargar el modelo ResNet50 preentrenado
from PIL import Image  # Para abrir y manejar imágenes
from scipy.spatial.distance import cosine  # Para calcular qué tan parecidas son las imágenes

from recortar_etiqueta_opencv import recortar_por_template  # Importo mi función de recorte automático de etiquetas

# ========================================
# 1. Defino cómo quiero transformar/preparar las imágenes antes de analizarlas
# ========================================

# Aquí especifico una serie de pasos que se aplican a cada imagen antes de pasarla al modelo.
# Estos pasos ayudan a que el modelo entienda bien las imágenes y que todas tengan el mismo formato.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiono todas las imágenes a 224x224 píxeles, que es el tamaño que espera el modelo
    transforms.ToTensor(),  # Convierto la imagen a tensor (estructura de datos que entiende PyTorch)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalizo los colores como se hizo durante el entrenamiento original
                         std=[0.229, 0.224, 0.225])
])

# ========================================
# 2. Creo una función para obtener un "vector de características" de una imagen
# ========================================

# Esta función carga una imagen, la transforma como indiqué arriba y la pasa por el modelo.
# El resultado es un vector (lista de números) que representa lo que "ve" el modelo en la imagen.
def get_image_embedding(img_path, model):
    img = Image.open(img_path).convert('RGB')  # Abro la imagen y me aseguro que esté en color (RGB)
    img_t = transform(img).unsqueeze(0)  # Le agrego una dimensión extra para simular un lote (batch size = 1)
    with torch.no_grad():  # Le indico a PyTorch que no quiero entrenar, solo predecir
        embedding = model(img_t).squeeze().numpy()  # Obtengo el vector de características y lo convierto a NumPy
    return embedding

# ========================================
# 3. Cargar el modelo de red neuronal ResNet50 y prepararlo
# ========================================

# Uso un modelo llamado ResNet50 ya entrenado con millones de imágenes (ImageNet)
# Elimino la última capa que clasifica, porque solo quiero el vector de características
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Me quedo solo con las capas que extraen características
resnet.eval()  # Pongo el modelo en modo "evaluación", no entrenamiento

# ========================================
# 4. Recortar automáticamente la etiqueta desde una imagen grande
# ========================================

# Aquí especifico las imágenes:
imagen_grande = 'Image/etiqueta_foto.jpg'         # Esta es la imagen que contiene la etiqueta incrustada en un entorno
imagen_base = 'Image/etiqueta_base_h.jpg'         # Esta es la imagen buena, la que uso como plantilla
imagen_recortada = 'Image/etiqueta_recortada.jpg' # Aquí voy a guardar el recorte detectado

# Llamo a la función de recorte para buscar y extraer la etiqueta automáticamente
recortar_por_template(imagen_grande, imagen_base, guardar_como=imagen_recortada)

# ========================================
# 5. Obtener los vectores de características de las imágenes
# ========================================

# Convierto la imagen base y la imagen recortada en vectores de características
embedding_base = get_image_embedding(imagen_base, resnet)
embedding_recortada = get_image_embedding(imagen_recortada, resnet)

# ========================================
# 6. Comparar las imágenes usando distancia coseno
# ========================================

# Calculo qué tan parecidos son los dos vectores
# Si son idénticos, la similitud será 1. Si son totalmente distintos, será 0
similaridad = 1 - cosine(embedding_base, embedding_recortada)
print(f"Similitud entre imagen base y recorte: {similaridad:.4f}")

# ========================================
# 7. Evaluar el resultado de la comparación
# ========================================

# Si la similitud es mayor a 0.85, considero que las imágenes son visualmente iguales (mismo tipo de etiqueta)
if similaridad > 0.85:
    print("✅ Las imágenes son visualmente del mismo tipo.")
else:
    print("❌ Las imágenes parecen ser diferentes.")