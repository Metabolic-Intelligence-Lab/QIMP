
from qiskit.circuit.library import CRYGate, HGate, RYGate
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib.gridspec import GridSpec
import imageio



def load_and_preprocess_images(image_paths):
    """
    Carica le immagini, applica filtri e calcola la GP.
    
    Args:
        image_paths (list): Percorsi delle immagini in formato RGB.
    
    Returns:
        tuple: Canale rosso, canale verde, immagine GP.
    """
    if len(image_paths) < 2:
        raise ValueError("Servono almeno due immagini RGB per calcolare la GP.")
    
    # Carica le immagini
    red_channel = np.array(Image.open(image_paths[0]).convert('L'), dtype=np.float32)
    green_channel = np.array(Image.open(image_paths[1]).convert('L'), dtype=np.float32)
    image_array = np.array(image, dtype=np.uint16)
    
    
    
    # Normalizzazione dei valori tra 0 e 1
    red_channel /= 4096.0
    green_channel /= 4096.0
    
    # Applicazione dei filtri
    filtered_red = gaussian_filter(red_channel, sigma=sigma)
    filtered_red = median_filter(filtered_red, size=size)
    filtered_green = gaussian_filter(green_channel, sigma=sigma)
    filtered_green = median_filter(filtered_green, size=size)
    
    # Ridimensiona le immagini a 16x16
    resized_red = np.array(Image.fromarray(filtered_red).resize((16, 16), Image.LANCZOS))
    resized_green = np.array(Image.fromarray(filtered_green).resize((16, 16), Image.LANCZOS))
    
    # Assicura che i valori siano nel range [0, 1] prima di calcolare arccos
    resized_red = np.clip(resized_red, 0, 1)
    resized_green = np.clip(resized_green, 0, 1)
    
    # Calcola l'immagine GP
    gp_array = (resized_green - 0.5 * resized_red) / (resized_green + 0.5 * resized_red + 1e-10)
    gp_array = np.clip(gp_array, -1, 1)
    
    return resized_red, resized_green, gp_array


def setup_quantum_circuit(n, m, angles_list):
    """
    Imposta il circuito quantistico con le immagini codificate.

    Args:
        n (int): Numero di qubit per dimensione.
        m (int): Numero di immagini.
        angles_list (list of np.ndarray): Lista di angoli codificati.

    Returns:
        QuantumCircuit: Circuito quantistico configurato.
    """
    n_qubits = 2 * n + m + 1
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    qc.h(range(n_qubits - 1))
    
    indices = [format(i, '0' + str(2 * n) + 'b')[::-1] for i in range(2 ** (2 * n))]
    selection_index = [format(j, '0' + str(m) + 'b')[::-1] for j in range(2**m)]
        
    for s_index, theta_values in enumerate(angles_list):
        
        for index, theta in zip(indices, theta_values):
            
            for bit_position, bit in enumerate(index):
                if bit == '0':
                    qc.x(bit_position)
                
            ry_gate = RYGate(theta).control(m + 2 * n)
            control_bits = list(range(n_qubits - 1)) + [n_qubits - 1]
            qc.append(ry_gate, control_bits)
                
            for bit_position, bit in enumerate(index):
                if bit == '0':
                    qc.x(bit_position)
            
            qc.barrier()

        for index2 in selection_index:
              
            for bit_position, bit in enumerate(index2):
                if bit == '0' and 2 * n + s_index < n_qubits:
                    qc.x(2 * n +s_index)
        qc.barrier()
    return qc

def decode_quantum_images(counts, n, m, normalization_factor):

    
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
    quantum_images = [[] for _ in range(num_images)]
    
    for index, value in sorted_pixel_values.items():
        image_index = int(index[:m], 2) if m > 0 else 0
        pixel_index = index[m:]
        row = int(pixel_index[:num_digits // 2], 2)
        col = int(pixel_index[num_digits // 2:], 2)
        if row >= image_size or col >= image_size:
            print(f"Index out of bounds: row {row}, col {col}, image_size {image_size}")
        else:
            quantum_images[image_index].append((row, col, value))
    
    decoded_images = []
    for pixel_values in quantum_images:
        image_quantum = np.zeros((image_size, image_size))
        for row, col, value in pixel_values:
            image_quantum[row, col] = value
        decoded_images.append(image_quantum)
    return decoded_images


def measure_quantum_circuit(qc):
    """
    Aggiunge le operazioni di misura al circuito quantistico.

    Args:
        qc (QuantumCircuit): Circuito quantistico da misurare.
    """
    n_qubits = qc.num_qubits
    qc.measure(range(n_qubits), range(n_qubits))


def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 1.0  # Supponendo che l'immagine sia normalizzata tra 0 e 1
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def total_variation(image):
    tv = np.sum(np.sqrt(np.diff(image, axis=0, append=image[-1:])**2 + np.diff(image, axis=1, append=image[:,-1:])**2))
    return tv


# Funzione per applicare filtri (senza soglia)
def apply_filters(channel_array):
    filtered_array = gaussian_filter(channel_array, sigma=sigma)
    filtered_array = median_filter(filtered_array, size=size)
    return filtered_array

# Funzione per calcolare l'immagine GP
def calculate_gp_image(green_channel, red_channel, G=0.5):
    gp_array = (green_channel - G * red_channel) / (green_channel + G * red_channel + 1e-10)
    gp_array = np.clip(gp_array, -1, 1)
    gp_array[green_channel + G * red_channel == 0] = 0
    return gp_array

def extract_index(bitstring):
    return int(bitstring, 2)

def apply_gp_function(qc, n, m, params):
    """
    Applica le operazioni quantistiche per calcolare (I1 - I2) / (I1 + I2).
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione spaziale.
        m (int): Numero di qubit per selezionare l'immagine.
        params (list of Parameter): Parametri del circuito da ottimizzare.
    """
    control_qubit = 2 * n
    color_qubit = 2 * n + m

    param_idx = 0
    for i in range(2 ** (2 * n)):
        binary_index = format(i, '0' + str(2 * n) + 'b')[::-1]
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        
        # Utilizzare parametri variabili per le rotazioni controllate
        qc.cry(params[param_idx], control_qubit, color_qubit)  # Operazione di differenza
        param_idx += 1
        # Simulare CH gate parametrizzato
        qc.append(CRYGate(params[param_idx]), [control_qubit, color_qubit])
        qc.append(HGate(), [color_qubit])
        param_idx += 1
        
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        qc.barrier()

def combined_objective(mse_value, psnr_value, tv_value, alpha=1, beta=1, gamma=1):
    return alpha * mse_value - beta * psnr_value + gamma * tv_value

def optimize_gp_function(n, m, gp_image_classic, angles_list, alpha=1, beta=1, gamma=1):
    """
    Ottimizza la funzione GP nel circuito quantistico per minimizzare l'MSE con l'immagine GP classica.
    
    Args:
        n (int): Numero di qubit per dimensione spaziale.
        m (int): Numero di qubit per selezionare l'immagine.
        gp_image_classic (np.ndarray): Immagine GP classica.
        angles_list (list): Lista di angoli per il setup iniziale del circuito.
        alpha (float): Peso per l'MSE.
        beta (float): Peso per il PSNR.
        gamma (float): Peso per la TV.
    
    Returns:
        QuantumCircuit: Circuito quantistico ottimizzato.
        list: Lista dei valori dell'MSE durante l'ottimizzazione.
        list: Lista dei valori del PSNR durante l'ottimizzazione.
        list: Lista dei valori della TV durante l'ottimizzazione.
        list: Lista dei valori della funzione obiettivo combinata durante l'ottimizzazione.
        list: Lista dei tempi ad ogni iterazione.
    """
    qc = setup_quantum_circuit(n, m, angles_list)
    params = [Parameter(f'θ{i}') for i in range(2 * (2 ** (2 * n)))]
    apply_gp_function(qc, n, m, params)
    measure_quantum_circuit(qc)
    mse_values = []
    psnr_values = []
    tv_values = []
    combined_values = []
    times = []
    start_time = time.time()
    
    def objective_function(values):
        bound_qc = qc.assign_parameters({param: value for param, value in zip(params, values)})
        simulator = AerSimulator()
        result = simulator.run(transpile(bound_qc, simulator), shots=10**(n+3)).result()
        counts = result.get_counts()
        quantum_images = decode_quantum_images(counts, n, m, 1)
        mse_value = mse(gp_image_classic, quantum_images[1])  # Calcola l'MSE rispetto alla seconda immagine quantistica
        psnr_value = psnr(gp_image_classic, quantum_images[1])  # Calcola il PSNR rispetto alla seconda immagine quantistica
        tv_value = total_variation(quantum_images[1])  # Calcola la TV rispetto alla seconda immagine quantistica
        combined_value = combined_objective(mse_value, psnr_value, tv_value, alpha, beta, gamma)
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        tv_values.append(tv_value)
        combined_values.append(combined_value)
        times.append(time.time() - start_time)
        print(f"Iteration: {len(mse_values)}, MSE: {mse_value}, PSNR: {psnr_value} dB, TV: {tv_value}, Combined: {combined_value}")
        return combined_value
    initial_params = np.random.uniform(-2 * np.pi, 2 * np.pi, size=len(params))
    result = minimize(objective_function, initial_params,method='COBYLA', options={'maxiter': 10,'disp': True})
    optimized_params = result.x
    optimized_qc = qc.assign_parameters({param: value for param, value in zip(params, optimized_params)})
    return optimized_qc, mse_values, psnr_values, tv_values, combined_values, times


def plot_images_strip(classic_images, quantum_strip, titles=None):
    num_classic = len(classic_images)
    num_quantum = len(quantum_strip)
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

def save_images_as_tiff(images, base_filename):
    """
    Salva una lista di immagini come file TIFF con nomi incrementali.

    Args:
        images (list of np.ndarray): Lista delle immagini da salvare.
        base_filename (str): Nome base del file da utilizzare per il salvataggio.
    """
    for i, image in enumerate(images):
        filename = f"{base_filename}_{i+1}.tif"
        imageio.imwrite(filename, image.astype(np.uint8), format='TIFF')
        print(f"Immagine salvata come {filename}")


def main(image_paths, alpha=1, beta=1, gamma=1):
    # Preprocessa le immagini e calcola la GP
    print("Caricamento e preprocessing delle immagini...")
    red_channel, green_channel, gp_image_classic = load_and_preprocess_images(image_paths)
    
    # Imposta i parametri per il circuito
    n = int(np.log2(red_channel.shape[0]))  # Calcola 'n' dalla dimensione
    m = 1  # Supponiamo un'immagine GP

    # Genera gli angoli per il circuito
    normalization_factor = 4096 / np.pi
    angles_list = [
        np.arccos(1 - 2 * red_channel / 4096).flatten(),
        np.arccos(1 - 2 * green_channel / 4096).flatten(),
    ]

    print("Ottimizzazione del circuito quantistico...")
    qc, mse_values, psnr_values, tv_values, combined_values, times = optimize_gp_function(
        n, m, gp_image_classic, angles_list, alpha, beta, gamma
    )

    # Visualizza il circuito ottimizzato
    print("Visualizzazione del circuito ottimizzato...")
    optimized_circuit_diagram = qc.draw(output='mpl')
    plt.show()

    print("Simulazione del circuito quantistico ottimizzato...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n+3))
    result = job.result()

    print("Recupero dei risultati...")
    counts = result.get_counts()
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)

    # Visualizza i risultati
    titles = [("Canale Verde", "Canale Rosso"), ("Immagine GP Classica", "Immagine GP Quantistica")]
    plot_images_strip([green_channel, red_channel, gp_image_classic], quantum_images, titles=titles)

    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, "quantum_image_output")

    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff([green_channel, red_channel, gp_image_classic], "classic_image_output")

    print("Fine del programma.")
    return optimized_circuit_diagram


# Parametri del filtro Gaussian e mediano
sigma = 1
size = 3

if __name__ == "__main__":
    # Percorsi delle immagini RGB
    image_paths = ["IMMAGINI PER QUANTUM/trainQML/Train_QML_16/membraneStack_Sample011_L_UV_DC_001rbc0DM2_ch01_16x16.tif", "IMMAGINI PER QUANTUM/trainQML/Train_QML_16/membraneStack_Sample011_L_UV_DC_001rbc0DM2_ch02_16x16.tif"]
    main(image_paths, alpha=0.001, beta=0.1, gamma=0.1)
