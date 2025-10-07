"""
Script para descargar imágenes de gatos y perros automáticamente
Requiere: pip install bing-image-downloader
"""

from bing_image_downloader import downloader
import os
import shutil

def download_dataset(num_images_per_class=100):
    """
    Descarga imágenes de gatos y perros
    """
    print("="*60)
    print("DESCARGANDO DATASET DE GATOS Y PERROS")
    print("="*60)
    
    # Crear carpetas temporales
    os.makedirs('temp_downloads', exist_ok=True)
    
    # Descargar gatos
    print(f"\n1. Descargando {num_images_per_class} imágenes de gatos...")
    downloader.download(
        "cat animal pet",
        limit=num_images_per_class,
        output_dir='temp_downloads',
        adult_filter_off=True,
        force_replace=False,
        timeout=15
    )
    
    # Descargar perros
    print(f"\n2. Descargando {num_images_per_class} imágenes de perros...")
    downloader.download(
        "dog animal pet",
        limit=num_images_per_class,
        output_dir='temp_downloads',
        adult_filter_off=True,
        force_replace=False,
        timeout=15
    )
    
    # Mover a carpetas correctas
    print("\n3. Organizando imágenes...")
    
    # Crear estructura de carpetas
    os.makedirs('images/cat', exist_ok=True)
    os.makedirs('images/dog', exist_ok=True)
    
    # Mover gatos
    cat_source = 'temp_downloads/cat animal pet'
    if os.path.exists(cat_source):
        for i, filename in enumerate(os.listdir(cat_source), 1):
            src = os.path.join(cat_source, filename)
            dst = os.path.join('images/cat', f'cat_{i:03d}.jpg')
            try:
                shutil.copy2(src, dst)
            except:
                pass
    
    # Mover perros
    dog_source = 'temp_downloads/dog animal pet'
    if os.path.exists(dog_source):
        for i, filename in enumerate(os.listdir(dog_source), 1):
            src = os.path.join(dog_source, filename)
            dst = os.path.join('images/dog', f'dog_{i:03d}.jpg')
            try:
                shutil.copy2(src, dst)
            except:
                pass
    
    # Limpiar carpeta temporal
    print("\n4. Limpiando archivos temporales...")
    shutil.rmtree('temp_downloads', ignore_errors=True)
    
    # Contar imágenes finales
    cat_count = len([f for f in os.listdir('images/cat') if f.endswith(('.jpg', '.jpeg', '.png'))])
    dog_count = len([f for f in os.listdir('images/dog') if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print("\n" + "="*60)
    print("✓ DESCARGA COMPLETADA")
    print("="*60)
    print(f"Gatos: {cat_count} imágenes en images/cat/")
    print(f"Perros: {dog_count} imágenes en images/dog/")
    print("\nAhora puedes ejecutar:")
    print("python create_features.py --samples cat images/cat/ --samples dog images/dog/ \\")
    print("  --codebook-file models/codebook.pkl --feature-map-file models/feature_map.pkl \\")
    print("  --num-clusters 150")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Descarga imágenes de gatos y perros')
    parser.add_argument('--num-images', type=int, default=100,
                       help='Número de imágenes por clase (default: 100)')
    args = parser.parse_args()
    
    try:
        download_dataset(args.num_images)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nAsegúrate de instalar: pip install bing-image-downloader")