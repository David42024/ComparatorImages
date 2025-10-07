"""
Script de diagnóstico para identificar problemas
"""
import pickle
import numpy as np
import os

print("="*60)
print("DIAGNÓSTICO DEL SISTEMA")
print("="*60)

# 1. Verificar imágenes
print("\n1. VERIFICANDO IMÁGENES:")
cat_files = [f for f in os.listdir('images/cat') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
dog_files = [f for f in os.listdir('images/dog') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"   Gatos: {len(cat_files)} imágenes")
print(f"   Perros: {len(dog_files)} imágenes")

if len(cat_files) == 0 or len(dog_files) == 0:
    print("   ✗ PROBLEMA: Una o ambas carpetas están vacías!")
else:
    print("   ✓ Ambas carpetas tienen imágenes")

# 2. Verificar feature map
print("\n2. VERIFICANDO FEATURE MAP:")
try:
    with open('models/feature_map.pkl', 'rb') as f:
        feature_map = pickle.load(f)
    
    print(f"   Total de muestras: {len(feature_map)}")
    
    if len(feature_map) == 0:
        print("   ✗ PROBLEMA: Feature map está vacío!")
    else:
        # Contar por clase
        labels = {}
        for item in feature_map:
            label = item['label']
            labels[label] = labels.get(label, 0) + 1
        
        print(f"   Distribución por clase:")
        for label, count in labels.items():
            print(f"     - {label}: {count} muestras")
        
        # Verificar vector de características
        sample = feature_map[0]
        print(f"\n   Muestra de ejemplo:")
        print(f"     - Label: {sample['label']}")
        print(f"     - Feature vector shape: {sample['feature_vector'].shape}")
        print(f"     - Feature vector type: {sample['feature_vector'].dtype}")
        
        # Verificar que no sean ceros
        vec_sum = np.sum(sample['feature_vector'])
        if vec_sum == 0:
            print(f"     ✗ PROBLEMA: Vector de características es todo ceros!")
        else:
            print(f"     ✓ Vector contiene datos (suma={vec_sum:.4f})")
        
        # Verificar todos los vectores
        zero_count = 0
        for item in feature_map:
            if np.sum(item['feature_vector']) == 0:
                zero_count += 1
        
        if zero_count > 0:
            print(f"   ✗ PROBLEMA: {zero_count}/{len(feature_map)} vectores son todo ceros!")
        else:
            print(f"   ✓ Todos los vectores contienen datos")

except FileNotFoundError:
    print("   ✗ PROBLEMA: No se encontró models/feature_map.pkl")
    print("   Debes ejecutar create_features.py primero")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# 3. Verificar codebook
print("\n3. VERIFICANDO CODEBOOK:")
try:
    with open('models/codebook.pkl', 'rb') as f:
        kmeans, centroids = pickle.load(f)
    
    print(f"   Número de clusters: {len(centroids)}")
    print(f"   Dimensión de centroides: {centroids.shape}")
    print(f"   ✓ Codebook cargado correctamente")
except FileNotFoundError:
    print("   ✗ PROBLEMA: No se encontró models/codebook.pkl")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

# 4. Probar extracción de características en imagen de prueba
print("\n4. PROBANDO EXTRACCIÓN DE CARACTERÍSTICAS:")
try:
    import cv2
    test_cat = os.path.join('images/cat', cat_files[0])
    img = cv2.imread(test_cat)
    
    if img is None:
        print(f"   ✗ PROBLEMA: No se pudo leer {test_cat}")
    else:
        print(f"   Imagen de prueba: {test_cat}")
        print(f"   Dimensiones: {img.shape}")
        
        # Extraer SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        if descriptors is None:
            print(f"   ✗ PROBLEMA: No se detectaron características SIFT!")
            print(f"   Keypoints detectados: {len(keypoints) if keypoints else 0}")
        else:
            print(f"   ✓ Keypoints detectados: {len(keypoints)}")
            print(f"   ✓ Descriptores extraídos: {descriptors.shape}")

except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n" + "="*60)
print("RECOMENDACIONES:")
print("="*60)

recommendations = []

if len(cat_files) < 20 or len(dog_files) < 20:
    recommendations.append("• Necesitas más imágenes (mínimo 30-50 por clase)")

if 'feature_map' in locals() and len(feature_map) < 40:
    recommendations.append("• Feature map tiene muy pocas muestras")

if 'zero_count' in locals() and zero_count > 0:
    recommendations.append("• Hay vectores vacíos - problema con extracción de características")
    recommendations.append("• Ejecuta: python create_features.py --samples cat images/cat/ --samples dog images/dog/ --codebook-file models/codebook.pkl --feature-map-file models/feature_map.pkl --num-clusters 150")

if not recommendations:
    recommendations.append("• Todo parece estar bien. El problema puede ser:")
    recommendations.append("  - Dataset muy pequeño")
    recommendations.append("  - Hiperparámetros de la red")
    recommendations.append("  - Necesitas más iteraciones de entrenamiento")

for rec in recommendations:
    print(rec)