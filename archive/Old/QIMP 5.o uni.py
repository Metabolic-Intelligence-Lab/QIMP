# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:18:50 2024

@author: Giuseppe
"""
from qiskit_aer import AerSimulator
from qiskit import transpile
from FQRI_lib2 import *
import matplotlib.pyplot as plt

def main(image_names):
    print("Caricamento delle immagini e codifica degli angoli...")
    angles_list, n, m, normalization_factor = load_and_encode_images(image_names)
    
    print("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit(n, m, angles_list)
    qc = rotate_90(qc, n)  # Esempio di rotazione a 90 gradi, opzionale
    measure_quantum_circuit(qc)
    
    print("Visualizzazione del circuito quantistico...")
    qc.draw('mpl')
    plt.show(fig)
    
    print("Simulazione del circuito quantistico...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n + 2))
    result = job.result()
    
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)
    
    print("Decodifica dei risultati...")
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)
    
    print("Visualizzazione delle immagini...")
    classic_images = [np.array(angles) * normalization_factor / np.pi for angles in angles_list]
    classic_images = [image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) for image in classic_images]
    
    titles = [(f"Immagine Classica {i+1}", f"Immagine Quantistica Ricostruita {i+1}") for i in range(len(image_names))]
    
    plot_images_strip(classic_images, quantum_images, titles=titles)
    
    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, "quantum_image_output")
    
    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff(classic_images, "classic_image_output")

if __name__ == "__main__":
    image_names = ["ratioch1_8-1.tif", "ratioch2_8-1.tif"]
    main(image_names)
