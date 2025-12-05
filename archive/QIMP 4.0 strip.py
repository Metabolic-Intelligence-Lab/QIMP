# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:16:31 2024

@author: Giuseppe
"""
from FQRI_lib import *
from qiskit_aer import AerSimulator
from qiskit import transpile


def main():
    image_names = ["ratioch1_16.tif"]#, "ratioch2_16.tif","ratioch2_16.tif","ratioch2_16.tif"]
    print("Caricamento delle immagini e codifica degli angoli...")
    angles_list, n , m= load_and_encode_images(image_names)
    
    # Determina il fattore di normalizzazione dall'immagine
    first_image_path = os.path.join(os.getcwd(), image_names[0])
    first_image = imageio.imread(first_image_path)
    normalization_factor = 4096# if first_image.max() > 255 else 255.0
    
   
    print("Impostazione del circuito quantistico per la quantum strip...")
    qc = setup_quantum_strip_circuit(n, m, angles_list)

    measure_quantum_circuit(qc)
    qc.draw()
  
    
  
    
    print("Simulazione del circuito quantistico...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n +1))
    result = job.result()

    
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)
    
    print("Decodifica dei risultati...")
    quantum_strip = decode_quantum_strip(counts, n, m, normalization_factor)
    
    print("Visualizzazione delle immagini...")
    classic_images = [np.array(angles) * normalization_factor / np.pi for angles in angles_list]
    classic_images = [image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) for image in classic_images]
    
    titles = [("Immagine Classica 1", "Immagine Quantistica Ricostruita 1"), 
              ("Immagine Classica 2", "Immagine Quantistica Ricostruita 2")]

    plot_images_strip(classic_images, quantum_strip, titles=titles)
    
    for i, image_quantum in enumerate(quantum_strip):
        print(f"Salvataggio dell'immagine quantistica {i}...")
        save_image_as_tiff(image_quantum, f"quantum_image_output_{i}.tif")

if __name__ == "__main__":
    main()
