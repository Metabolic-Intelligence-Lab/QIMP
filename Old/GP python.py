import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit import transpile
from FQRI_lib2 import *
from skimage import exposure
from PIL import Image

def calculate_average_image(quantum_images):
    """
    Calcola l'immagine media dalle immagini quantistiche ricostruite.
    
    Args:
        quantum_images (list): Lista di immagini quantistiche ricostruite.
    
    Returns:
        np.ndarray: Immagine media.
    """
    average_image = np.mean(quantum_images, axis=0)
    return average_image

def calculate_classical_gp_image(image1, image2):
    """
    Calcola l'immagine GP classica usando la formula (I1 - I2) / (I1 + I2).
    
    Args:
        image1 (np.ndarray): Prima immagine.
        image2 (np.ndarray): Seconda immagine.
    
    Returns:
        np.ndarray: Immagine GP classica.
    """
    gp_classical = (image1 - 0.05*image2) / (image1 + 0.05*image2) +1e-8 # Aggiungiamo una piccola costante per evitare divisioni per zero
    return gp_classical

def normalize_image(image):
    """
    Normalizza un'immagine scalando i valori di pixel tra 0 e 1.
    
    Args:
        image (np.ndarray): Immagine da normalizzare.
    
    Returns:
        np.ndarray: Immagine normalizzata.
    """
    norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return norm_image

def enhance_contrast(image):
    """
    Migliora il contrasto di un'immagine usando l'equalizzazione dell'istogramma.
    
    Args:
        image (np.ndarray): Immagine da migliorare.
    
    Returns:
        np.ndarray: Immagine con contrasto migliorato.
    """
    enhanced_image = exposure.equalize_hist(image)
    return enhanced_image

def plot_images_comparison(quantum_average_image, classical_gp_image):
    """
    Grafica l'immagine media quantistica e l'immagine GP classica per confronto.
    
    Args:
        quantum_average_image (np.ndarray): Immagine media quantistica.
        classical_gp_image (np.ndarray): Immagine GP classica.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im0 = axes[0].imshow(quantum_average_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Quantum Average Image')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(classical_gp_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Classical GP Image')
    fig.colorbar(im1, ax=axes[1])
    
    plt.show()

def main(image_names):
    print("Caricamento delle immagini e codifica degli angoli...")
    angles_list, n, m, normalization_factor = load_and_encode_images(image_names)
    
    print("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit(n, m, angles_list)
    qc = calculate_gp_quantum_circuit(qc, n, m)
    measure_quantum_circuit(qc)
    qc.draw()
    
    print("Simulazione del circuito quantistico...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n + 1))
    result = job.result()
    
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)
    
    print("Decodifica dei risultati...")
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)
    
    print("Visualizzazione delle immagini...")
    classic_images = [np.array(angles) * normalization_factor / np.pi for angles in angles_list]
    classic_images = [image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) for image in classic_images]
    
    titles = [(f"Immagine Classica {i+1}",
               f"Immagine Quantistica Ricostruita {i+1}")
              for i in range(len(image_names))]
    
    plot_images_strip(classic_images, quantum_images, titles=titles)
    """
    for i, image_quantum in enumerate(quantum_images):
        print(f"Salvataggio dell'immagine quantistica {i}...")
        save_image_as_tiff(image_quantum, f"quantum_image_output_{i}.tif")
    """
    # Calcolo delle immagini medie e visualizzazione
    quantum_average_image = calculate_average_image(quantum_images)
    classical_gp_image = calculate_classical_gp_image(classic_images[0], classic_images[1])
    
    # Normalizzazione delle immagini
    quantum_average_image_normalized = normalize_image(quantum_average_image)
    classical_gp_image_normalized = normalize_image(classical_gp_image)
    
    # Miglioramento del contrasto dell'immagine GP classica
    classical_gp_image_enhanced = enhance_contrast(classical_gp_image_normalized)
    
    print("Visualizzazione dell'immagine media quantistica e dell'immagine GP classica normalizzate e con contrasto migliorato...")
    plot_images_comparison(quantum_average_image_normalized, classical_gp_image_enhanced )

if __name__ == "__main__":
    image_names = ["ratioch1_16.tif", "ratioch2_16.tif"]
    main(image_names)
