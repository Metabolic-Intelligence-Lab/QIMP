#!/usr/bin/env python3
"""
Quick test per generare immagini GP
"""

import os
from glob import glob
import numpy as np
from PIL import Image
from quantum_utils import calculate_gp_image

def quick_test():
    """Test veloce di generazione GP"""
    print("🧪 Quick GP Generation Test")
    
    # Verifica directory
    input_dir = 'IMMAGINI PER QUANTUM/trainQML'
    output_dir = os.path.join(input_dir, 'GP_output')
    
    print(f"📁 Input dir: {input_dir}")
    print(f"📁 Output dir: {output_dir}")
    print(f"📁 Input exists: {os.path.exists(input_dir)}")
    
    # Crea output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory")
    
    # Trova immagini
    tif_files = glob(os.path.join(input_dir, "*.tif"))
    print(f"🔍 Found {len(tif_files)} TIF files")
    
    if not tif_files:
        print("❌ No TIF files found!")
        return
    
    # Test con prime 2 immagini
    for i, img_file in enumerate(tif_files[:2]):
        try:
            print(f"\n🔄 Processing {os.path.basename(img_file)}")
            
            # Carica immagine
            img = Image.open(img_file)
            img_array = np.array(img, dtype=np.float32)
            
            print(f"  📊 Shape: {img_array.shape}")
            print(f"  📊 Dtype: {img_array.dtype}")
            print(f"  📊 Range: [{img_array.min():.1f}, {img_array.max():.1f}]")
            
            # Controlla se ha canali multipli
            if len(img_array.shape) < 3:
                print("  ⚠️ Immagine senza canali multipli, skip")
                continue
                
            if img_array.shape[2] < 2:
                print("  ⚠️ Non abbastanza canali per GP, skip")
                continue
            
            # Estrai primi 2 canali
            ch1 = img_array[:, :, 0]  # Red
            ch2 = img_array[:, :, 1]  # Green
            
            print(f"  📊 Ch1 (Red) range: [{ch1.min():.1f}, {ch1.max():.1f}]")
            print(f"  📊 Ch2 (Green) range: [{ch2.min():.1f}, {ch2.max():.1f}]")
            
            # Ridimensiona a 16x16
            ch1_pil = Image.fromarray(ch1.astype(np.uint16))
            ch2_pil = Image.fromarray(ch2.astype(np.uint16))
            
            ch1_16 = np.array(ch1_pil.resize((16, 16), Image.Resampling.LANCZOS))
            ch2_16 = np.array(ch2_pil.resize((16, 16), Image.Resampling.LANCZOS))
            
            # Calcola GP
            gp_image = calculate_gp_image(ch2_16.astype(np.float32), ch1_16.astype(np.float32), 
                                        G=0.5, output_format='16bit')
            
            print(f"  📊 GP range: [{gp_image.min()}, {gp_image.max()}]")
            
            # Salva GP
            base_name = os.path.basename(img_file).replace('.tif', '')
            gp_output = os.path.join(output_dir, f"{base_name}_GP.tif")
            
            Image.fromarray(gp_image).save(gp_output)
            print(f"  ✅ Saved: {gp_output}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Verifica risultati
    gp_files = glob(os.path.join(output_dir, "*_GP.tif"))
    print(f"\n🎯 Generated {len(gp_files)} GP files")
    for gp_file in gp_files:
        print(f"  📄 {os.path.basename(gp_file)}")

if __name__ == "__main__":
    quick_test()