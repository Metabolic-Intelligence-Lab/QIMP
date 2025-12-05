#!/usr/bin/env python3
"""
Test script per verificare che le correzioni al path funzionino
"""

import os
import sys

def test_path_resolution():
    """Test the path resolution logic"""
    print("🧪 Testing Path Resolution...")
    
    # Test paths (same logic as in the fixed script)
    input_directory = os.path.join('IMMAGINI PER QUANTUM', 'trainQML', 'Train_QML_16')
    output_directory = os.path.join('IMMAGINI PER QUANTUM', 'trainQML', 'GP_output')
    fallback_input = os.path.join('IMMAGINI PER QUANTUM', 'trainQML')
    fallback_output = '.'
    
    print(f"  Input directory: {input_directory}")
    print(f"  Exists: {os.path.exists(input_directory)}")
    
    # Check if directories exist (same logic as fixed script)
    if not os.path.exists(input_directory):
        print(f"  ⚠️ Directory {input_directory} non trovata, usando fallback...")
        image_directory = fallback_input
        if not os.path.exists(image_directory):
            print(f"  ❌ Fallback directory anche mancante: {image_directory}")
            return False
    else:
        image_directory = input_directory
    
    if not os.path.exists(output_directory):
        print(f"  ⚠️ Directory output {output_directory} non trovata, uso directory corrente")
        output_dir = fallback_output
    else:
        output_dir = output_directory
    
    print(f"  📁 Final input directory: {image_directory}")
    print(f"  📁 Final output directory: {output_dir}")
    
    # Test file detection
    try:
        all_files = os.listdir(image_directory)
        input_files = [os.path.join(image_directory, file) for file in all_files 
                      if file.endswith('.tif') and 'ch' in file.lower()]
        
        if os.path.exists(output_dir):
            output_files = [os.path.join(output_dir, file) for file in os.listdir(output_dir) 
                           if file.endswith('.tif') and 'GP' in file]
        else:
            output_files = []
            
        print(f"  🔍 Trovati {len(input_files)} file di input, {len(output_files)} file di output")
        
        if len(input_files) > 0:
            print(f"  📄 Primi file di input: {[os.path.basename(f) for f in input_files[:3]]}")
            return True
        else:
            print(f"  ❌ Nessun file di input trovato!")
            return False
            
    except Exception as e:
        print(f"  ❌ Errore lettura directory: {e}")
        return False

def test_import():
    """Test that the fixed script can be imported"""
    print("\n🧪 Testing Script Import...")
    
    try:
        # Read the script but don't execute main
        with open('quantize_analyze_quantum_gp_v2.py', 'r') as f:
            script_content = f.read()
        
        # Remove the main execution part
        script_lines = script_content.split('\n')
        filtered_lines = []
        skip_main = False
        
        for line in script_lines:
            if 'if __name__ == "__main__":' in line:
                skip_main = True
                continue
            if not skip_main:
                filtered_lines.append(line)
        
        # Try to compile
        filtered_script = '\n'.join(filtered_lines)
        compile(filtered_script, 'test_script', 'exec')
        print("  ✅ Script compila correttamente")
        return True
        
    except Exception as e:
        print(f"  ❌ Errore compilazione: {e}")
        return False

if __name__ == "__main__":
    print("Testing Fixed Quantize and Analyze Script")
    print("=" * 50)
    
    test1_passed = test_path_resolution()
    test2_passed = test_import()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("Tutti i test passati! Lo script dovrebbe funzionare ora.")
        print("\nEsempio di esecuzione:")
        print("   python quantize_analyze_quantum_gp_v2.py")
    else:
        print("Alcuni test falliti. Verificare i problemi sopra.")
    
    print("\nRisultati:")
    print("  Path resolution: " + ("OK" if test1_passed else "KO"))
    print("  Script compilation: " + ("OK" if test2_passed else "KO"))
