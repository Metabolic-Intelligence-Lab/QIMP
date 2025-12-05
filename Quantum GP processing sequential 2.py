
from qiskit.circuit.library import CRYGate, HGate, RYGate
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
import imageio
from glob import glob
import os

# ---------------------------
# Funzioni per il preprocessing
# ---------------------------
def apply_filters(channel_array):
    filtered_array = gaussian_filter(channel_array, sigma=sigma)
    filtered_array = median_filter(filtered_array, size=size)
    return filtered_array

def calculate_gp_image(green_channel, red_channel, G=0.5):
    # Invertito l'ordine dei canali rosso e verde
    gp_array = (green_channel - G * red_channel) / (green_channel + G * red_channel + 1e-10)
    gp_array = np.clip(gp_array, -1, 1)
    gp_array[green_channel + G * red_channel == 0] = 0
    gp_image_16bit = np.interp(gp_array, [-1, 1], [0, 4096]).astype(np.uint16)
    return gp_image_16bit

def visualize_channels_and_gp(green_channel, red_channel, gp_image):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(green_channel, cmap='Greens')
    axs[0].set_title('Canale Verde')
    axs[1].imshow(red_channel, cmap='Reds')
    axs[1].set_title('Canale Rosso')
    axs[2].imshow(gp_image, cmap='gray')
    axs[2].set_title('Immagine GP')
    plt.show()







def load_and_preprocess_classical_images(image_path):
    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.uint16)
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    filtered_red_channel = apply_filters(red_channel)
    filtered_green_channel = apply_filters(green_channel)
    resized_red_channel = np.array(Image.fromarray(filtered_red_channel).convert('L').resize((16, 16), Image.LANCZOS))
    resized_green_channel = np.array(Image.fromarray(filtered_green_channel).convert('L').resize((16, 16), Image.LANCZOS))
    gp_classical = calculate_gp_image(resized_green_channel, resized_red_channel)
    visualize_channels_and_gp(resized_green_channel, resized_red_channel, gp_classical)
    print("GP Classica Range:", gp_classical.min(), gp_classical.max())
    return resized_red_channel, resized_green_channel, gp_classical

def plot_images_strip(classic_images, quantum_images, titles):
    num_classic = len(classic_images)
    num_quantum = len(quantum_images)
    num_cols = max(num_classic, num_quantum)
    
    fig, axes = plt.subplots(2, num_cols, figsize=(15, 10))
    for i in range(num_classic):
        axes[0, i].imshow(classic_images[i], cmap='gray')
        if i < len(titles):
            axes[0, i].set_title(titles[i])
        axes[0, i].axis("off")
    for j in range(num_quantum):
        if quantum_images[j] is not None:
            axes[1, j].imshow(quantum_images[j], cmap='gray')
            if num_classic + j < len(titles):
                axes[1, j].set_title(titles[num_classic + j])
            axes[1, j].axis("off")
        else:
            print(f"Errore: Immagine quantistica {j+1} non caricata correttamente.")
    plt.tight_layout()
    plt.show()

# ---------------------------
# Funzioni per la costruzione dei circuiti
# ---------------------------
def setup_quantum_circuit(n, m, angles_list):
    """
    Costruisce il circuito fisso in funzione dei valori dei pixel.
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
                # Se l'intenzione è controllare i singoli qubit, si potrebbe usare bit_position:
                if bit == '0' and 2 * n + s_index < n_qubits:
                    qc.x(2 * n + s_index)
            qc.barrier()
    return qc

def apply_gp_function(qc, n, m, params):
    """
    Aggiunge al circuito qc la parte parametrizzata che esegue la funzione GP.
    I parametri passati in 'params' non verranno ricreati ad ogni immagine.
    """
    control_qubit = 2 * n
    color_qubit = 2 * n + m
    param_idx = 0
    for i in range(2 ** (2 * n)):
        binary_index = format(i, '0' + str(2 * n) + 'b')[::-1]
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        qc.cry(params[param_idx], control_qubit, color_qubit)  # Rotazione controllata
        param_idx += 1
        qc.append(CRYGate(params[param_idx]), [control_qubit, color_qubit])
        qc.append(HGate(), [color_qubit])
        param_idx += 1
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        qc.barrier()

def measure_quantum_circuit(qc):
    n_qubits = qc.num_qubits
    qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------
# Funzioni per la valutazione degli errori
# ---------------------------
def mse(imageA, imageB):
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def total_variation(image):
    tv = np.sum(np.sqrt(np.diff(image, axis=0, append=image[-1:])**2 +
                        np.diff(image, axis=1, append=image[:,-1:])**2))
    return tv

def combined_objective(mse_value, psnr_value, tv_value, alpha=1, beta=1, gamma=1):
    return alpha * mse_value - beta * psnr_value + gamma * tv_value

def extract_index(bitstring):
    return int(bitstring, 2)

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

def save_images_as_tiff(images, base_filename):
    for i, image in enumerate(images):
        filename = f"{base_filename}_{i+1}.tif"
        imageio.imwrite(filename, image.astype(np.uint8), format='TIFF')
        print(f"Immagine salvata come {filename}")

# ---------------------------
# MAIN: training su più immagini senza reinizializzare i parametri della funzione GP
# ---------------------------
# Parametri del filtro
sigma = 1
size = 3
input_directory = 'IMMAGINI PER QUANTUM/trainQML'
image_files = glob(os.path.join(input_directory, '*.tif'))


def infer_quantum_image(image_path, optimized_params, n, m, alpha=1, beta=1, gamma=1):
    """
    Esegue un'inferenza su una nuova immagine utilizzando il circuito ottimizzato.
    
    Parameters:
    - image_path: percorso dell'immagine da inferire.
    - optimized_params: i parametri ottimizzati dal training.
    - n: dimensione dell'immagine (es. log2 della dimensione dell'immagine GP).
    - m: numero di immagini.
    - alpha, beta, gamma: pesi per l'obiettivo combinato (opzionali).
    
    Returns:
    - quantum_image: l'immagine quantistica risultante.
    """
    # Carica e preprocessa l'immagine
    red_channel, green_channel, gp_image_classic = load_and_preprocess_classical_images(image_path)
    
    # Impostazione del circuito quantistico
    angles_list = [
        np.arccos(1 - 2 * red_channel / 4096).flatten(),
        np.arccos(1 - 2 * green_channel / 4096).flatten()
    ]
    
    # Creazione del circuito fisso
    fixed_circuit = setup_quantum_circuit(n, m, angles_list)
    
    # Composizione del circuito con quello parametrizzato
    num_qubits = 2 * n + m + 1
    parametrized_subcircuit = QuantumCircuit(num_qubits)
    apply_gp_function(parametrized_subcircuit, n, m, [Parameter(f'θ{i}') for i in range(2 * (2 ** (2 * n)))])

    # Applicazione dei parametri ottimizzati
    bound_circuit = fixed_circuit.compose(parametrized_subcircuit, qubits=range(num_qubits))
    bound_circuit = bound_circuit.assign_parameters({param: value for param, value in zip([Parameter(f'θ{i}') for i in range(2 * (2 ** (2 * n)))], optimized_params)})

    # Misurazione del circuito
    measure_quantum_circuit(bound_circuit)

    # Esegui il circuito sul simulatore
    shots = 10 ** (n + 3)
    simulator = AerSimulator()
    job = simulator.run(transpile(bound_circuit, simulator), shots=shots)
    result_sim = job.result()

    # Decodifica l'immagine quantistica
    counts = result_sim.get_counts()
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor=4096 / np.pi)

    # Visualizza l'immagine quantistica risultante
    plt.figure(figsize=(6, 6))
    plt.imshow(quantum_images[0], cmap='gray')
    plt.title("Inferenza Immagine Quantistica")
    plt.axis('off')
    plt.show()

    return quantum_images[0]  # Restituisce l'immagine quantistica risultante


# ---------------------------
# Funzione di inferenza
# ---------------------------
def infer_quantum_image(image_path, optimized_params, n, m):
    red_channel, green_channel, gp_image_classic = load_and_preprocess_classical_images(image_path)
    angles_list = [
        np.arccos(1 - 2 * red_channel / 4096).flatten(),
        np.arccos(1 - 2 * green_channel / 4096).flatten()
    ]
    num_qubits = 2 * n + m + 1
    qc = setup_quantum_circuit(n, m, angles_list)
    apply_gp_function(qc, n, m, optimized_params)
    measure_quantum_circuit(qc)
    simulator = AerSimulator(device='gpu')
    job = simulator.run(transpile(qc, simulator), shots=10 ** (n + 3))
    result_sim = job.result()
    counts = result_sim.get_counts()
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor=4096 / np.pi)
    quantum_gp_image = quantum_images[0]
    mse_value = mse(gp_image_classic, quantum_gp_image)
    psnr_value = psnr(gp_image_classic, quantum_gp_image)
    tv_value = total_variation(quantum_gp_image)
    print(f"MSE: {mse_value}, PSNR: {psnr_value}, TV: {tv_value}")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(gp_image_classic, cmap='gray')
    plt.title("GP Classica")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(quantum_gp_image, cmap='gray')
    plt.title("GP Quantistica")
    plt.axis("off")
    plt.show()
    return quantum_gp_image, mse_value, psnr_value, tv_value


def main_sequential(input_directory, alpha=1, beta=1, gamma=1):
    image_files = glob(os.path.join(input_directory, '*.tif'))
    print(f"Trovate {len(image_files)} immagini da processare in sequenza.")
    
    simulator = AerSimulator()
    
    # Fase di setup globale per la parte parametrizzata (usando la prima immagine per definire n e m)
    if len(image_files) == 0:
        print("Nessuna immagine trovata.")
        return
    red_channel, green_channel, gp_image_classic = load_and_preprocess_classical_images(image_files[0])
    n = int(np.log2(gp_image_classic.shape[0]))
    m = 1  # Modifica se necessario
    num_params = 2 * (2 ** (2 * n))
    
    # Parametri globali che non verranno reinizializzati ad ogni immagine
    global_params = [Parameter(f'θ{i}') for i in range(num_params)]
    optimized_params = np.random.uniform(-2 * np.pi, 2 * np.pi, size=num_params)
    
    num_qubits = 2 * n + m + 1
    parametrized_subcircuit = QuantumCircuit(num_qubits)
    apply_gp_function(parametrized_subcircuit, n, m, global_params)
    parametrized_subcircuit.barrier()
    
    global_error_history = []  # Per tracciare l'errore globale
    
    for idx, file in enumerate(image_files):
        print(f"\nElaborazione dell'immagine {idx+1}/{len(image_files)}: {file}")
        try:
            red_channel, green_channel, gp_image_classic = load_and_preprocess_classical_images(file)
        except Exception as e:
            print(f"Errore nel preprocessamento dell'immagine {file}: {e}")
            continue
        
        n = int(np.log2(gp_image_classic.shape[0]))
        m = 1
        
        angles_list = [
            np.arccos(1 - 2 * red_channel / 4096).flatten(),
            np.arccos(1 - 2 * green_channel / 4096).flatten()
        ]
        
        fixed_circuit = setup_quantum_circuit(n, m, angles_list)
        full_circuit = fixed_circuit.compose(parametrized_subcircuit, qubits=range(num_qubits))
        measure_quantum_circuit(full_circuit)
        
        shots = 10 ** (n + 4                                                                                         )
        
        # Definisci error_history per questa immagine
        error_history = []
        
        def objective_function(values):
            bound_circuit = full_circuit.assign_parameters({param: value for param, value in zip(global_params, values)})
            result = simulator.run(transpile(bound_circuit, simulator), shots=shots).result()
            counts = result.get_counts()
            quantum_images = decode_quantum_images(counts, n, m, normalization_factor=4096/np.pi)
            mse_value = mse(gp_image_classic, quantum_images[0])
            psnr_value = psnr(gp_image_classic, quantum_images[0])
            tv_value = total_variation(quantum_images[0])
            combined_value = mse_value #combined_objective(mse_value, psnr_value, tv_value, alpha, beta, gamma)
            error_history.append(combined_value)
            global_error_history.append(combined_value)
            print(f"    Errore combinato: {combined_value}")
            return combined_value
        
        # Ottimizzazione per l'immagine corrente
        result_opt = minimize(objective_function, optimized_params, method='COBYLA', options={'maxiter': 50, 'disp': True})
        optimized_params = result_opt.x  # Aggiorna per il prossimo giro
        
        bound_circuit = full_circuit.assign_parameters({param: value for param, value in zip(global_params, optimized_params)})
        bound_circuit = transpile(bound_circuit, simulator)
        job = simulator.run(bound_circuit, shots=shots)
        result_sim = job.result()
        counts = result_sim.get_counts()
        quantum_images = decode_quantum_images(counts, n, m, normalization_factor=4096/np.pi)
        
        titles = [("Canale Verde", "Canale Rosso", "Immagine GP Classica"), "Immagine GP Quantistica"]
        plot_images_strip([green_channel, red_channel, gp_image_classic], quantum_images, titles=titles)
        base_filename = f"quantum_image_output_{os.path.splitext(os.path.basename(file))[0]}"
        save_images_as_tiff(quantum_images, base_filename)
        print(f"Immagine {idx+1} processata e salvata.\n")
        
        # Plotta l'andamento dell'errore per questa immagine
        plt.figure()
        plt.plot(error_history, marker='o', linestyle='-')
        plt.xlabel("Iterazioni")
        plt.ylabel("Errore Combinato")
        plt.title(f"Andamento dell'errore per immagine {idx+1}")
        plt.grid(True)
        plt.show()
    
    # Plotta l'andamento globale dell'errore
    plt.figure()
    plt.plot(global_error_history, marker='o', linestyle='-')
    plt.xlabel("Iterazioni totali")
    plt.ylabel("Errore Combinato")
    plt.title("Andamento globale dell'errore nel training")
    plt.grid(True)
    plt.show()
    print("Elaborazione sequenziale completata.")
    
    
    
    for file in image_files:
       infer_quantum_image(file, optimized_params, n, m)
    print("Inferenza su nuove immagini")
    test_image_files = glob(os.path.join(input_directory, 'test/*.tif'))
    for test_file in test_image_files:
        print(f"Inferenza su {test_file}")
        quantum_gp_image, mse_value, psnr_value, tv_value = infer_quantum_image(test_file, optimized_params, n, m)
        print(f"Metriche per {test_file} -> MSE: {mse_value}, PSNR: {psnr_value}, TV: {tv_value}")
    print("Inferenza completata.")
    
    
    
    print("Elaborazione sequenziale completata.")


# Chiamata al main
if __name__ == "__main__":
    input_directory = 'IMMAGINI PER QUANTUM/trainQML'
    main_sequential(input_directory, alpha=1, beta=1, gamma=1)
