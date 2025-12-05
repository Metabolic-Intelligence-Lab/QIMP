# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:35:40 2024

@author: Giuseppe
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:18:50 2024

@author: Giuseppe
"""

from qiskit_aer import AerSimulator
from qiskit import transpile
from FQRI_lib2 import *
from qiskit.circuit import Parameter

def calculate_mad(image1, image2):
    return np.mean(np.abs(image1 - image2))




def calculate_gp(qc, n, m):
    n_qubits = 2 * n + m + 1
    ancilla_qubit = n_qubits - 1

    # Loop over all possible combinations of qubits in the strip
    for i in range(2 ** (2 * n)):
        binary = format(i, f'0{2 * n}b')
        for j in range(2 ** m):
            control_qubits = list(range(2 * n)) + [2 * n + j]
            control_qubits = list(set(control_qubits))  # Ensure no duplicates
            if ancilla_qubit in control_qubits:
                control_qubits.remove(ancilla_qubit)  # Ensure ancilla_qubit is not in control_qubits

            if len(control_qubits) > 0:  # Check if control_qubits list is not empty
                qc.mcx(control_qubits, ancilla_qubit)
                theta = np.pi / 2
                qc.rz(theta, ancilla_qubit)
                qc.mcx(control_qubits, ancilla_qubit)

    # Subtract I2 from I1
    qc.x(ancilla_qubit)
    
    return qc

def renormalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)  # Normalizzazione tra 0 e 1
    return normalized_image


def calculate_classical_gp(image1, image2):
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    gp = (image1 - image2) / (image1 + image2 + 1e-8)  # Aggiungi una piccola costante per evitare divisioni per zero
    gp = np.clip(gp, -1, 1)  # Clipping per assicurarsi che i valori siano tra -1 e 1
    return gp

def calculate_similarity(classical_gp, quantum_gp):
    """
    Calcola la somiglianza tra la GP classica e quella quantistica utilizzando la Differenza Assoluta Media (MAD).
    
    Args:
        classical_gp (np.ndarray): Immagine della GP classica.
        quantum_gp (np.ndarray): Immagine della GP quantistica.
    
    Returns:
        float: Differenza Assoluta Media (MAD) tra le due immagini.
    """
    if classical_gp.shape != quantum_gp.shape:
        raise ValueError("Le dimensioni delle immagini GP classica e quantistica devono essere uguali.")
    
    mad = np.mean(np.abs(classical_gp - quantum_gp))
    return mad

# Esempio di utilizzo nella funzione main
def main(image_names):
    print("Caricamento delle immagini e codifica degli angoli...")
    angles_list, n, m, normalization_factor = load_and_encode_images(image_names)
    
    print("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit(n, m, angles_list)
    
    print("Calcolo del GP...")
    qc = calculate_gp(qc, n, m)
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
    
    print("Calcolo classico del GP...")
    classic_images = [np.array(angles) * normalization_factor / np.pi for angles in angles_list]
    classic_images = [image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) for image in classic_images]
    classical_gp = calculate_classical_gp(classic_images[0], classic_images[1])
    
    print("Rinormalizzazione del GP classico...")
    classical_gp_normalized = renormalize_image(classical_gp)
    
    print("Visualizzazione delle immagini...")
    titles = [(f"Immagine Classica {i+1}", f"Immagine Quantistica Ricostruita {i+1}") for i in range(len(image_names))]
    plot_images_strip(classic_images, quantum_images, titles=titles)
    
    print("Visualizzazione del confronto tra calcolo classico e quantistico...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(classical_gp_normalized, cmap='gray')
    axes[0].set_title("Immagine GP Classica Rinormalizzata")
    axes[0].axis('off')
    
    if len(quantum_images) > 0:
        axes[1].imshow(quantum_images[0], cmap='gray')
        axes[1].set_title("Immagine GP Quantistica")
        axes[1].axis('off')
    
    plt.show()
    
    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, "quantum_image_output")
    
    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff(classic_images, "classic_image_output")
    save_images_as_tiff([classical_gp_normalized], "classical_gp_output")
    
    print("Calcolo della somiglianza tra GP classica e quantistica...")
    if len(quantum_images) > 0:
        quantum_gp = quantum_images[0]  # Assumi che la prima immagine quantistica sia quella corrispondente
        similarity_score = calculate_similarity(classical_gp_normalized, quantum_gp)
        print(f"Differenza Assoluta Media (MAD) tra GP classica e quantistica: {similarity_score}")

if __name__ == "__main__":
    image_names = ["ratioch1_16.tif", "ratioch2_16.tif"]
    main(image_names)
