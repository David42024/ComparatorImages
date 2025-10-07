"""
Script para aumentar el dataset mediante transformaciones
"""
import cv2
import os
import numpy as np
from pathlib import Path

def augment_image(img):
    """
    Aplica transformaciones a una imagen
    Retorna lista de imágenes aumentadas
    """
    augmented = []
    
    # Flip horizontal
    augmented.append(('flip', cv2.flip(img, 1)))
    
    # Rotaciones
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    for angle in [-15, -10, 10, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append((f'rot{angle}', rotated))
    
    # Brillo
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    augmented.append(('bright', bright))
    
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
    augmented.append(('dark', dark))
    
    # Zoom (recorte central + resize)
    crop_percent = 0.85
    crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
    start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
    zoomed = cv2.resize(cropped, (w, h))
    augmented.append(('zoom', zoomed))
    
    # Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    augmented.append(('blur', blurred))
    
    return augmented

def augment_dataset(input_folder, output_folder, target_count=50):
    """
    Aumenta un dataset completo
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Obtener imágenes existentes
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    original_count = len(image_files)
    print(f"  Imágenes originales: {original_count}")
    
    if original_count == 0:
        print(f"  ✗ ERROR: No hay imágenes en {input_folder}")
        return
    
    if original_count >= target_count:
        print(f"  ⚠ Ya tienes {original_count} imágenes (>= {target_count})")
        print(f"  Copiando todas a {output_folder}...")
        for img_file in image_files:
            src = os.path.join(input_folder, img_file)
            dst = os.path.join(output_folder, img_file)
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)
        return
    
    # Copiar y aumentar
    print(f"  Generando {target_count - original_count} imágenes adicionales...")
    
    count = 0
    
    # Primero copiar originales
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            base_name = Path(img_file).stem
            ext = Path(img_file).suffix
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_orig{ext}"), img)
            count += 1
    
    # Luego aumentar hasta llegar al target
    img_index = 0
    aug_round = 0
    
    while count < target_count:
        img_file = image_files[img_index % len(image_files)]
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            img_index += 1
            continue
        
        base_name = Path(img_file).stem
        augmented_images = augment_image(img)
        
        # Guardar aumentaciones
        for aug_type, aug_img in augmented_images:
            if count >= target_count:
                break
            
            output_path = os.path.join(output_folder, f"{base_name}_{aug_type}_r{aug_round}.jpg")
            cv2.imwrite(output_path, aug_img)
            count += 1
        
        img_index += 1
        if img_index % len(image_files) == 0:
            aug_round += 1
    
    final_count = len([f for f in os.listdir(output_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  ✓ Dataset aumentado: {original_count} → {final_count} imágenes")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Aumenta dataset de imágenes')
    parser.add_argument('--category', required=True, choices=['cat', 'dog', 'both'],
                       help='Categoría a aumentar')
    parser.add_argument('--target', type=int, default=50,
                       help='Número objetivo de imágenes por categoría')
    args = parser.parse_args()
    
    print("="*60)
    print("DATA AUGMENTATION")
    print("="*60)
    
    if args.category in ['cat', 'both']:
        print("\n📸 Procesando GATOS...")
        augment_dataset('images/cat', 'images/cat_augmented', args.target)
    
    if args.category in ['dog', 'both']:
        print("\n📸 Procesando PERROS...")
        augment_dataset('images/dog', 'images/dog_augmented', args.target)
    
    print("\n" + "="*60)
    print("✓ COMPLETADO")
    print("="*60)
    print("\n📋 SIGUIENTE PASO:")
    print("="*60)
    
    if args.category == 'both':
        print("python create_features.py --samples cat images/cat_augmented/ \\")
        print("  --samples dog images/dog_augmented/ --codebook-file models/codebook.pkl \\")
        print("  --feature-map-file models/feature_map.pkl --num-clusters 150")