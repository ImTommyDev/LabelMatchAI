# ========================================
# Script auxiliar para detectar y recortar automáticamente la etiqueta de una imagen grande
# utilizando OpenCV. Lo que hago aquí es comparar una imagen buena (la plantilla) con una imagen
# completa que contiene más contenido, para encontrar y recortar solo la etiqueta que me interesa.
# ========================================

import cv2
import numpy as np
from PIL import Image

# ========================================
# 1. Función para detectar y recortar la etiqueta con diferentes tamaños (escalas)
# ========================================

def recortar_por_template(imagen_grande_path, imagen_plantilla_path, guardar_como=None):
    """
    Esta función la uso para encontrar automáticamente una etiqueta dentro de una imagen más grande.

    - Lo que hago es comparar una imagen "base" (la buena, la plantilla) con otra imagen más grande
      que contiene más cosas (por ejemplo, la caja completa o una mesa).
    
    - Pruebo a buscar la plantilla en la imagen grande en diferentes tamaños, porque a veces la etiqueta
      puede estar más pequeña o más grande dependiendo de la distancia o la cámara.
    
    - Una vez encuentro la coincidencia más parecida, recorto justo esa zona de la imagen grande.
    
    - Si le paso el parámetro `guardar_como`, también guardo el recorte como archivo en el disco.
    
    - Devuelvo el recorte como una imagen PIL, para que pueda usarla o enseñarla después.
    """

    # Primero convierto ambas imágenes (la grande y la plantilla) a escala de grises.
    # Así elimino los colores y me centro solo en formas y estructuras, que es lo que usa OpenCV para comparar.
    imagen_gris = cv2.imread(imagen_grande_path, cv2.IMREAD_GRAYSCALE)
    plantilla_original = cv2.imread(imagen_plantilla_path, cv2.IMREAD_GRAYSCALE)

    # Si alguna imagen no se pudo cargar, lanzo un error para saber que hay un problema con las rutas
    if imagen_gris is None or plantilla_original is None:
        raise FileNotFoundError("No se pudieron cargar las imágenes. Verifica las rutas.")

    # Aquí voy a guardar la mejor coincidencia que encuentre
    mejor_valor = -1
    mejor_top_left = None
    mejor_escala = 1.0
    mejor_h, mejor_w = plantilla_original.shape[:2]

    # Ahora hago un bucle donde pruebo diferentes tamaños de la plantilla (desde 50% hasta 150%)
    # Esto es útil porque la etiqueta puede estar más pequeña o más grande en la imagen grande
    for escala in np.linspace(0.5, 1.5, 20):
        plantilla_redimensionada = cv2.resize(plantilla_original, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

        # Si la plantilla escalada se vuelve más grande que la imagen grande, la salto
        if plantilla_redimensionada.shape[0] > imagen_gris.shape[0] or plantilla_redimensionada.shape[1] > imagen_gris.shape[1]:
            continue

        # Aquí comparo la plantilla escalada con la imagen grande para ver dónde encaja mejor
        resultado = cv2.matchTemplate(imagen_gris, plantilla_redimensionada, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(resultado)

        # Me quedo con la mejor coincidencia encontrada hasta el momento
        if max_val > mejor_valor:
            mejor_valor = max_val
            mejor_top_left = max_loc
            mejor_escala = escala
            mejor_h, mejor_w = plantilla_redimensionada.shape[:2]

    # Si no encontré ninguna coincidencia, aviso de que algo ha ido mal
    if mejor_top_left is None:
        raise ValueError("No se encontró una coincidencia adecuada entre plantilla e imagen.")

    # Estas son las coordenadas del área a recortar
    top_left = mejor_top_left
    bottom_right = (top_left[0] + mejor_w, top_left[1] + mejor_h)

    # Cargo la imagen original en color (no en blanco y negro) y recorto solo la parte que necesito
    imagen_color = cv2.imread(imagen_grande_path)
    recorte = imagen_color[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Convierto el recorte a una imagen de tipo PIL (que es más fácil de manejar con Python)
    recorte_pil = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))

    # Si me han dicho dónde guardar el recorte, lo guardo como imagen en disco
    if guardar_como:
        recorte_pil.save(guardar_como)

    # Imprimo por pantalla qué escala usé y qué tan buena fue la coincidencia
    print(f"Recorte realizado con escala {mejor_escala:.2f}. Similitud máxima encontrada: {mejor_valor:.4f}")
    return recorte_pil

# ========================================
# Ejemplo de uso (descomentar para probar el script directamente)
# ========================================

# if __name__ == "__main__":
#     recorte = recortar_por_template(
#         'Image/etiqueta_foto.jpg',           # Imagen general que contiene la etiqueta
#         'Image/etiqueta_base_h.jpg',         # Imagen buena que actúa como plantilla
#         guardar_como='Image/etiqueta_recortada.jpg'  # Ruta de guardado del recorte
#     )
#     recorte.show()  # Mostrar el recorte en pantalla para verificar que quedó bien
