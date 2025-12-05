#!/usr/bin/env python3
"""
Generatore Ottimizzato di Immagini GP

Script migliorato per generare immagini GP dalle immagini multi-canale.
Integra le ottimizzazioni di quantum_utils.py.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from quantum_utils import (
    calculate_gp_image, apply_filters_optimized, 
    visualize_channels_and_gp, performance_monitor,
    get_memory_pool, batch_process_images
)

# Configurazione directories
INPUT_DIRECTORY = 'IMMAGINI PER QUANTUM/trainQML'
OUTPUT_DIRECTORY = os.path.join(INPUT_DIRECTORY, 'GP_output')
TRAIN_QML_16_DIRECTORY = os.path.join(INPUT_DIRECTORY, 'Train_QML_16')

# Parametri ottimizzati
TARGET_SIZE = (16, 16)
SIGMA = 1.0
MEDIAN_SIZE = 3
BATCH_SIZE = 10

def setup_directories():
    """Crea le directory necessarie se non esistono"""
    for directory in [OUTPUT_DIRECTORY, TRAIN_QML_16_DIRECTORY]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Creata directory: {directory}")

@performance_monitor
def process_single_image(image_file):
    """
    Processa una singola immagine multi-canale per generare GP e canali separati.
    
    Args:
        image_file: Path al file immagine
        
    Returns:
        Dict con risultati del processing
    """
    try:
        print(f"🔄 Processing: {os.path.basename(image_file)}")
        
        # Carica immagine
        image = Image.open(image_file)
        image_array = np.array(image, dtype=np.uint16)
        
        # Verifica che abbia almeno 2 canali (per GP)
        if len(image_array.shape) < 3 or image_array.shape[2] < 2:
            print(f"⚠️ Immagine {image_file} non ha abbastanza canali per GP")
            return {"success": False, "reason": "insufficient_channels"}
        
        # Estrai canali (assumendo Red=0, Green=1)
        red_channel = image_array[:, :, 0].astype(np.float32)
        green_channel = image_array[:, :, 1].astype(np.float32)
        
        # Applica filtri ottimizzati
        memory_pool = get_memory_pool(image_size=max(red_channel.shape), max_images=4)
        buffer1 = memory_pool.get_image_buffer()
        buffer2 = memory_pool.get_image_buffer()
        
        filtered_red = apply_filters_optimized(red_channel, SIGMA, MEDIAN_SIZE, buffer1[:red_channel.shape[0], :red_channel.shape[1]])
        filtered_green = apply_filters_optimized(green_channel, SIGMA, MEDIAN_SIZE, buffer2[:green_channel.shape[0], :green_channel.shape[1]])
        
        # Ridimensiona a 16x16 usando PIL per qualità migliore
        red_pil = Image.fromarray(filtered_red.astype(np.uint16))
        green_pil = Image.fromarray(filtered_green.astype(np.uint16))
        
        resized_red_pil = red_pil.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        resized_green_pil = green_pil.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        resized_red = np.array(resized_red_pil, dtype=np.float32)
        resized_green = np.array(resized_green_pil, dtype=np.float32)
        
        # Salva canali separati 16x16
        base_name = os.path.basename(image_file).replace('.tif', '')
        red_output_file = os.path.join(TRAIN_QML_16_DIRECTORY, f"{base_name}_ch01_16x16.tif")
        green_output_file = os.path.join(TRAIN_QML_16_DIRECTORY, f"{base_name}_ch02_16x16.tif")
        
        resized_red_pil.save(red_output_file)
        resized_green_pil.save(green_output_file)
        
        # Calcola GP usando funzione ottimizzata
        gp_image = calculate_gp_image(resized_green, resized_red, G=0.5, output_format='16bit')
        
        # Salva immagine GP
        gp_output_file = os.path.join(OUTPUT_DIRECTORY, f"{base_name}_GP.tif")
        Image.fromarray(gp_image).save(gp_output_file)
        
        return {
            "success": True,
            "files_generated": {
                "red_channel": red_output_file,
                "green_channel": green_output_file,
                "gp_image": gp_output_file
            },
            "gp_stats": {
                "min": float(gp_image.min()),
                "max": float(gp_image.max()),
                "mean": float(gp_image.mean())
            }
        }
        
    except Exception as e:
        print(f"❌ Errore processing {image_file}: {e}")
        return {"success": False, "reason": str(e)}

@performance_monitor 
def process_images_batch(image_files, show_progress=True):
    """
    Processa un batch di immagini in modo ottimizzato.
    
    Args:
        image_files: Lista di path alle immagini
        show_progress: Se mostrare progress durante processing
        
    Returns:
        Dict con statistiche del processing
    """
    results = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "gp_images_created": 0,
        "channel_pairs_created": 0,
        "errors": []
    }
    
    print(f"🚀 Inizio processing di {len(image_files)} immagini...")
    
    for i, image_file in enumerate(image_files):
        if show_progress and (i + 1) % 10 == 0:
            print(f"📊 Progress: {i + 1}/{len(image_files)} ({(i+1)/len(image_files)*100:.1f}%)")
        
        result = process_single_image(image_file)
        results["total_processed"] += 1
        
        if result["success"]:
            results["successful"] += 1
            results["gp_images_created"] += 1
            results["channel_pairs_created"] += 1
        else:
            results["failed"] += 1
            results["errors"].append({
                "file": image_file,
                "reason": result["reason"]
            })
    
    return results

def find_input_images():
    """Trova tutte le immagini .tif nella directory di input"""
    tif_pattern = os.path.join(INPUT_DIRECTORY, "*.tif")
    image_files = glob(tif_pattern)
    
    # Filtra solo immagini che non sono già processate
    filtered_files = []
    for img_file in image_files:
        base_name = os.path.basename(img_file)
        if not ("_ch01_" in base_name or "_ch02_" in base_name or "_GP" in base_name):
            filtered_files.append(img_file)
    
    print(f"🔍 Trovate {len(filtered_files)} immagini da processare")
    return filtered_files

def check_existing_outputs():
    """Controlla quanti file GP sono già stati generati"""
    gp_files = glob(os.path.join(OUTPUT_DIRECTORY, "*_GP.tif"))
    ch_files = glob(os.path.join(TRAIN_QML_16_DIRECTORY, "*_ch*_16x16.tif"))
    
    print(f"📄 File esistenti:")
    print(f"  - Immagini GP: {len(gp_files)}")
    print(f"  - Canali 16x16: {len(ch_files)}")
    
    return len(gp_files), len(ch_files)

@performance_monitor
def main(force_regenerate=False, show_visualizations=False, max_images=None):
    """
    Funzione principale per generare immagini GP.
    
    Args:
        force_regenerate: Se rigenerare anche file esistenti
        show_visualizations: Se mostrare plot durante processing
        max_images: Limite massimo di immagini da processare (None = tutte)
    """
    print("🎯 Generatore Ottimizzato di Immagini GP")
    print("=" * 50)
    
    # Setup
    setup_directories()
    
    # Controlla stato esistente
    existing_gp, existing_channels = check_existing_outputs()
    
    # Trova immagini da processare
    input_images = find_input_images()
    
    if not input_images:
        print("✅ Nessuna nuova immagine da processare!")
        return
    
    # Limita numero se richiesto
    if max_images is not None:
        input_images = input_images[:max_images]
        print(f"🔢 Limitato a {max_images} immagini per test")
    
    # Filtra file già processati se non force_regenerate
    if not force_regenerate:
        to_process = []
        for img_file in input_images:
            base_name = os.path.basename(img_file).replace('.tif', '')
            gp_file = os.path.join(OUTPUT_DIRECTORY, f"{base_name}_GP.tif")
            if not os.path.exists(gp_file):
                to_process.append(img_file)
        input_images = to_process
        
        if not input_images:
            print("✅ Tutte le immagini GP sono già state generate!")
            return
        
        print(f"🔄 {len(input_images)} immagini nuove da processare")
    
    # Processing
    results = process_images_batch(input_images)
    
    # Report finale
    print("\n" + "=" * 50)
    print("📊 RISULTATI FINALI")
    print(f"  ✅ Successi: {results['successful']}")
    print(f"  ❌ Fallimenti: {results['failed']}")
    print(f"  🖼️ Immagini GP create: {results['gp_images_created']}")
    print(f"  📱 Coppie canali create: {results['channel_pairs_created']}")
    
    if results['errors']:
        print(f"\n⚠️ Errori ({len(results['errors'])}):")
        for error in results['errors'][:5]:  # Show max 5 errors
            print(f"  - {os.path.basename(error['file'])}: {error['reason']}")
        if len(results['errors']) > 5:
            print(f"  ... e altri {len(results['errors']) - 5} errori")
    
    # Check finale
    final_gp, final_channels = check_existing_outputs()
    print(f"\n🎯 STATO FINALE:")
    print(f"  📄 Immagini GP totali: {final_gp}")
    print(f"  📱 Canali 16x16 totali: {final_channels}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Genera immagini GP ottimizzate")
    parser.add_argument("--force", action="store_true", help="Rigenera anche file esistenti")
    parser.add_argument("--viz", action="store_true", help="Mostra visualizzazioni durante processing")
    parser.add_argument("--max", type=int, help="Numero massimo di immagini da processare (per test)")
    
    args = parser.parse_args()
    
    main(
        force_regenerate=args.force,
        show_visualizations=args.viz,
        max_images=args.max
    )