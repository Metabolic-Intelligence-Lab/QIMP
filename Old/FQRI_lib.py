import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import math
from qiskit.circuit.library import RYGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from matplotlib.gridspec import GridSpec




# --- Section 1: Image Loading and Encoding ---

def load_and_encode_images(image_names):
    """
    Carica immagini e le codifica in angoli per l'input nei circuiti quantistici.
    
    Args:
        image_names (list): Lista dei nomi dei file delle immagini.
    
    Returns:
        angles_list (list): Lista degli angoli codificati per ogni immagine.
        n (int): Numero di qubit per dimensione.
        m (int): Numero di immagini meno uno.
    """
    angles_list = []
    n = None
    m = int(math.log2(len(image_names)))
    
    for image_name in image_names:
        working_dir = os.getcwd()
        image_path = os.path.join(working_dir, image_name)
        image = imageio.imread(image_path)
        
        #if len(image.shape) == 3:
        #    image = image[:, :, 0]
        
        max_value = image.max()
        normalization_factor = 4096.0 if max_value > 255 else 255.0
        
        image = image / normalization_factor
        image = np.clip(1 - 2 * image, -1, 1)
        angles = np.arccos(image)
        angles_flat = angles.flatten()
        
        angles_list.append(angles_flat)
        
        if n is None:
            num_pixels = angles_flat.size
            n = int(math.log2(math.sqrt(num_pixels)))
    
    return angles_list, n, m

def image_to_angles(image_name):
    """
    Carica una singola immagine e la codifica in angoli.
    
    Args:
        image_name (str): Nome del file dell'immagine.
    
    Returns:
        angles_flat (numpy.ndarray): Angoli codificati.
        n (int): Numero di qubit per dimensione.
    """
    working_dir = os.getcwd()
    image_path = os.path.join(working_dir, image_name)
    image = imageio.imread(image_path)
    
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    image = image / 4096
    image = np.clip(1 - 2 * image, -1, 1)
    angles = np.arccos(image)
    angles_flat = angles.flatten()
    
    num_pixels = angles_flat.size
    n = int(math.log2(math.sqrt(num_pixels)))
    
    return angles_flat, n

# --- Section 2: Quantum Circuit Setup ---

def setup_quantum_strip_circuit(n, m, angles_list):
    """
    Imposta un circuito quantistico per codificare più immagini.
    
    Args:
        n (int): Numero di qubit per dimensione.
        m (int): Numero di immagini meno uno.
        angles_list (list): Lista degli angoli codificati per ogni immagine.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con le immagini codificate.
    """
    n_qubits = 2 * n + m + 1
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    qc.h(range(n_qubits - 1))
    
    indices = [format(i, '0' + str(n_qubits - 1) + 'b')[::-1] for i in range(2 ** (n_qubits - 1))]
    
    for s_index, theta_values in enumerate(angles_list):
        for index, theta in zip(indices, theta_values):
            for bit_position, bit in enumerate(index):
                if bit == '0':
                    qc.x(bit_position)
                    
            ry_gate = RYGate(theta).control(n_qubits - 1)
            control_bits = list(range(n_qubits - 1)) + [n_qubits - 1]
            qc.append(ry_gate, control_bits)
            
            for bit_position, bit in enumerate(index):
                if bit == '0':
                    qc.x(bit_position)
            
            qc.barrier()
        
        if 2 * n + s_index < n_qubits:
            qc.x(2 * n + s_index)

        
        
        qc.barrier()
            
    image_qubit = qc.num_qubits - 2
    qc.x(image_qubit)
    return qc

def setup_quantum_circuit(n, theta_values):
    """
    Imposta un circuito quantistico per codificare una singola immagine.
    
    Args:
        n (int): Numero di qubit per dimensione.
        theta_values (numpy.ndarray): Angoli codificati.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con l'immagine codificata.
    """
    n_qubits = 2 * n + 1
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits - 1))
    
    indices = [format(i, '0' + str(n_qubits - 1) + 'b')[::-1] for i in range(2 ** (n_qubits - 1))]
    for index, theta in zip(indices, theta_values):
        for bit_position, bit in enumerate(index):
            if bit == '0':
                qc.x(bit_position)
        ry_gate = RYGate(theta).control(n_qubits - 1)
        qc.append(ry_gate, list(range(n_qubits - 1)) + [n_qubits - 1])
        for bit_position, bit in enumerate(index):
            if bit == '0':
                qc.x(bit_position)
        qc.barrier()
    
    return qc

# --- Section 3: Quantum Circuit Measurement and Processing ---

def measure_quantum_circuit(qc):
    """
    Misura il circuito quantistico.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico da misurare.
    """
    n_qubits = qc.num_qubits
    qc.measure(range(n_qubits), range(n_qubits))

def neutralize_color(qc):
    """
    Applica una porta Hadamard all'ultimo qubit per neutralizzare il colore.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con il colore neutralizzato.
    """
    last_qubit = qc.num_qubits - 1
    qc.h(last_qubit)
    return qc

def apply_zz_gate(qc, qubit1, qubit2, theta):
    """
    Applica una porta ZZ tra due qubit.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        qubit1 (int): Indice del primo qubit.
        qubit2 (int): Indice del secondo qubit.
        theta (float): Angolo di rotazione.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con la porta ZZ applicata.
    """
    qc.cx(qubit1, qubit2)
    qc.rz(theta, qubit2)
    qc.cx(qubit1, qubit2)
    return qc

def apply_x_on_coordinate(qc, n, coord):
    """
    Applica una porta X sui qubit di coordinata.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione.
        coord (str): Coordinata ('x' o 'y').
    """
    if coord == 'x':
        for qubit in range(n):
            qc.x(qubit)
    elif coord == 'y':
        for qubit in range(n, 2 * n):
            qc.x(qubit)
    else:
        raise ValueError("Coordinate non riconosciute. Usare 'x' o 'y'.")

def apply_coordinate_swapping(qc, n):
    """
    Applica lo scambio di coordinate tra x e y.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione.
    """
    for i in range(n):
        x_qubit = i
        y_qubit = n + i
        qc.swap(x_qubit, y_qubit)

def rotate_90(qc, n):
    """
    Ruota l'immagine di 90 gradi.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con l'immagine ruotata di 90 gradi.
    """
    apply_coordinate_swapping(qc, n)
    apply_x_on_coordinate(qc, n, 'x')
    return qc

def rotate_180(qc, n):
    """
    Ruota l'immagine di 180 gradi.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con l'immagine ruotata di 180 gradi.
    """
    apply_x_on_coordinate(qc, n, 'y')
    apply_x_on_coordinate(qc, n, 'x')
    return qc

def rotate_270(qc, n):
    """
    Ruota l'immagine di 270 gradi.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con l'immagine ruotata di 270 gradi.
    """
    apply_coordinate_swapping(qc, n)
    apply_x_on_coordinate(qc, n, 'y')
    return qc

def apply_color_inversion(qc):
    """
    Inverte i colori dell'immagine.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con i colori invertiti.
    """
    last_qubit = qc.num_qubits - 1
    qc.x(last_qubit)
    return qc

def apply_color_transformation(qc):
    """
    Applica una porta Z all'ultimo qubit per la trasformazione del colore.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con la trasformazione del colore.
    """
    last_qubit = qc.num_qubits - 1
    qc.z(last_qubit)
    return qc

def apply_custom_color_transformation(qc, theta):
    """
    Applica una trasformazione di colore personalizzata utilizzando una matrice unitaria.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        theta (float): Angolo di rotazione per la trasformazione del colore.
    
    Returns:
        QuantumCircuit: Il circuito quantistico con la trasformazione del colore applicata.
    """
    cos_theta = np.cos(theta / 2)
    sin_theta = np.sin(theta / 2)
    unitary_matrix = np.array([[cos_theta, sin_theta], [sin_theta, -cos_theta]])
    last_qubit = qc.num_qubits - 1
    custom_gate = Operator(unitary_matrix)
    qc.unitary(custom_gate, [last_qubit], label=f"C(2*{theta})")
    return qc

# --- Section 4: Quantum Image Decoding ---

def decode_quantum_strip(counts, n, m, normalization_factor):
    """
    Decodifica il circuito quantistico in immagini utilizzando i risultati delle misure.
    
    Args:
        counts (dict): Risultati delle misure del circuito quantistico.
        n (int): Numero di qubit per dimensione.
        m (int): Numero di immagini meno uno.
        normalization_factor (float): Fattore di normalizzazione per l'immagine.
    
    Returns:
        list: Lista di immagini decodificate.
    """
    num_pixels_per_image = 2 ** (2 * n)
    num_images = 2 ** m
    total_shots = sum(counts.values())
    prob = {k: v / total_shots for k, v in counts.items()}
    
    prob_cond_0 = {}
    prob_cond_1 = {}
    values_q = {}
    
    for state, p in prob.items():
        j = state[1:]
        if state[0] == '0':
            prob_cond_0[j] = p
        else:
            prob_cond_1[j] = p
    
    for j in set(prob_cond_0.keys()) | set(prob_cond_1.keys()):
        p_0 = prob_cond_0.get(j, 0)
        p_1 = prob_cond_1.get(j, 0)
        if p_0 + p_1 > 0:
            values_q[j] = np.arccos(np.sqrt(p_1 / (p_0 + p_1))) * normalization_factor * (2 / np.pi)
        else:
            values_q[j] = 0
    
    num_digits = len(next(iter(values_q.keys())))
    sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: extract_index(item[0]))}
    
    image_size = int(np.sqrt(num_pixels_per_image))
    quantum_strip = [[] for _ in range(num_images)]
    
    for index, value in sorted_pixel_values.items():
        image_index = int(index[:m], 2)
        pixel_index = index[m:]
        row = int(pixel_index[:num_digits // 2], 2)
        col = int(pixel_index[num_digits // 2:], 2)
        if row >= image_size or col >= image_size:
            print(f"Index out of bounds: row {row}, col {col}, image_size {image_size}")
        else:
            quantum_strip[image_index].append((row, col, value))
    
    decoded_images = []
    for pixel_values in quantum_strip:
        image_quantum = np.zeros((image_size, image_size))
        for row, col, value in pixel_values:
            image_quantum[row, col] = value
        decoded_images.append(image_quantum)
    
    return decoded_images

def decode_quantum_image(counts, n):
    """
    Decodifica una singola immagine quantistica dai risultati delle misure.
    
    Args:
        counts (dict): Risultati delle misure del circuito quantistico.
        n (int): Numero di qubit per dimensione.
    
    Returns:
        numpy.ndarray: Immagine quantistica decodificata.
    """
    num_pixels = 2 ** (2 * n)
    total_shots = sum(counts.values())
    prob = {k: v / total_shots for k, v in counts.items()}
    
    prob_cond_0 = {}
    prob_cond_1 = {}
    values_q = {}
    
    for state, p in prob.items():
        j = state[1:]
        if state[0] == '0':
            prob_cond_0[j] = p
        else:
            prob_cond_1[j] = p
    
    for j in set(prob_cond_0.keys()) | set(prob_cond_1.keys()):
        p_0 = prob_cond_0.get(j, 0)
        p_1 = prob_cond_1.get(j, 0)
        if p_0 + p_1 > 0:
            values_q[j] = np.arccos(np.sqrt(p_0 / (p_0 + p_1))) * 255 * (2 / np.pi)
        else:
            values_q[j] = 0
    
    num_digits = len(next(iter(values_q.keys())))
    sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: extract_index(item[0]))}
    image_quantum = np.zeros((int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))))
    
    for index, value in sorted_pixel_values.items():
        row = int(index[:num_digits // 2], 2)
        col = int(index[num_digits // 2:], 2)
        image_quantum[row][col] = value
    
    return image_quantum

# --- Section 5: Image Visualization and Saving ---

def plot_images_strip(classic_images, quantum_strip, titles=None):
    """
    Visualizza le immagini classiche e quantistiche in una singola striscia.
    
    Args:
        classic_images (list): Lista di immagini classiche.
        quantum_strip (list): Lista di immagini quantistiche.
        titles (list): Lista di tuple (titolo_classico, titolo_quantistico) per ogni coppia di immagini.
    """
    num_classic = len(classic_images)
    num_quantum = len(quantum_strip)
    total_images = num_classic + num_quantum
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, max(num_classic, num_quantum), figure=fig)
    
    for i in range(num_classic):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(classic_images[i], cmap='gray')
        ax.axis('off')
        title = titles[i][0] if titles and len(titles) > i else "Immagine Classica"
        ax.set_title(title)
    
    for j in range(num_quantum):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(quantum_strip[j], cmap='gray')
        ax.axis('off')
        title = titles[j][1] if titles and len(titles) > j else "Immagine Quantistica"
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def plot_images(classic_images, quantum_images, titles=None):
    """
    Visualizza immagini classiche e quantistiche affiancate.
    
    Args:
        classic_images (list): Lista di immagini classiche.
        quantum_images (list): Lista di immagini quantistiche.
        titles (list): Lista di tuple (titolo_classico, titolo_quantistico) per ogni coppia di immagini.
    """
    num_images = len(classic_images)
    for i in range(num_images):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(classic_images[i], cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        plt.title(titles[i][0] if titles else "Immagine Classica")
        
        plt.subplot(1, 2, 2)
        plt.imshow(quantum_images[i], cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        plt.title(titles[i][1] if titles else "Immagine Quantistica Ricostruita")
        plt.show()

def save_image_as_tiff(image_array, file_name):
    """
    Salva un array di immagini come file TIFF.
    
    Args:
        image_array (numpy.ndarray): Array dell'immagine da salvare.
        file_name (str): Nome del file in cui salvare l'immagine.
    """
    imageio.imwrite(file_name, image_array.astype(np.uint8), format='TIFF')

# --- Utility Functions ---

def extract_index(bitstring):
    """
    Converte una stringa binaria in un indice intero.
    
    Args:
        bitstring (str): Stringa binaria.
    
    Returns:
        int: Indice intero.
    """
    return int(bitstring, 2)

