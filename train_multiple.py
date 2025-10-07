import subprocess
import os
import shutil

def train_and_evaluate(iteration):
    """Entrena y retorna la precisión promedio"""
    print(f"\n{'='*60}")
    print(f"ITERACIÓN {iteration}")
    print('='*60)
    
    # Entrenar
    result = subprocess.run(
        ['python', 'training.py', 
         '--feature-map-file', 'models/feature_map.pkl',
         '--training-set', '0.8',
         '--ann-file', f'models/temp_ann_{iteration}.yaml',
         '--le-file', f'models/temp_le_{iteration}.pkl'],
        capture_output=True,
        text=True
    )
    
    # Extraer precisión del output
    output = result.stdout
    accuracies = []
    
    for line in output.split('\n'):
        if '%' in line and any(word in line for word in ['cat', 'dog']):
            try:
                acc_str = line.split('\t')[-1].strip().replace('%', '')
                accuracies.append(float(acc_str))
            except:
                pass
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    print(output)
    print(f"\nPrecisión promedio: {avg_accuracy:.2f}%")
    
    return avg_accuracy, f'models/temp_ann_{iteration}.yaml', f'models/temp_le_{iteration}.pkl'

if __name__ == '__main__':
    num_iterations = 10
    best_accuracy = 0
    best_ann_file = None
    best_le_file = None
    
    results = []
    
    for i in range(1, num_iterations + 1):
        accuracy, ann_file, le_file = train_and_evaluate(i)
        results.append((i, accuracy, ann_file, le_file))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_ann_file = ann_file
            best_le_file = le_file
    
    # Guardar el mejor modelo
    if best_ann_file:
        shutil.copy(best_ann_file, 'models/ann.yaml')
        shutil.copy(best_le_file, 'models/le.pkl')
        
        # Limpiar archivos temporales
        for i, _, ann_file, le_file in results:
            try:
                os.remove(ann_file)
                os.remove(le_file)
            except:
                pass
        
        print(f"\n{'='*60}")
        print("RESUMEN DE RESULTADOS")
        print('='*60)
        for i, acc, _, _ in results:
            marker = " ← MEJOR" if acc == best_accuracy else ""
            print(f"Iteración {i:2d}: {acc:6.2f}%{marker}")
        
        print(f"\n{'='*60}")
        print(f"Mejor precisión: {best_accuracy:.2f}%")
        print(f"Modelo guardado en: models/ann.yaml")
        print('='*60)