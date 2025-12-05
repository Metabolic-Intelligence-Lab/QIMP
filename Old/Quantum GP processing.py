
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

def apply_filters(channel_array):
    filtered_array = gaussian_filter(channel_array, sigma=sigma)
    filtered_array = median_filter(filtered_array, size=size)
    return filtered_array

# Funzione per calcolare l'immagine GP
def calculate_gp_image(green_channel, red_channel, G=0.5):
    # Invertito l'ordine dei canali rosso e verde
    gp_array = (green_channel - G * red_channel) / (green_channel + G * red_channel + 1e-10)
    gp_array = np.clip(gp_array, -1, 1)

    # Imposta lo sfondo a zero
    gp_array[green_channel + G * red_channel == 0] = 0

    # Interpolazione per 16-bit
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

    # Converti l'immagine in un array NumPy a 16-bit per gestire valori fino a 65535
    image_array = np.array(image, dtype=np.uint16)

    # Controlla che ci siano almeno 3 canali

    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    

    
    filtered_red_channel = apply_filters(red_channel)
    filtered_green_channel = apply_filters(green_channel)
  
    # Ridimensiona i canali a 16x16
    resized_red_channel = np.array(Image.fromarray(filtered_red_channel).convert('L').resize((16, 16), Image.LANCZOS))
    resized_green_channel = np.array(Image.fromarray(filtered_green_channel).convert('L').resize((16, 16), Image.LANCZOS))
  
  
    # Calcola l'immagine GP
    gp_classical = calculate_gp_image(resized_green_channel, resized_red_channel)  # Inverto red e green

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


def extract_index(bitstring):
    return int(bitstring, 2)

def apply_gp_function(qc, n, m, params):
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
    result = minimize(objective_function, initial_params,method='COBYLA', options={'maxiter': 3,'disp': True})
    optimized_params = result.x
    optimized_qc = qc.assign_parameters({param: value for param, value in zip(params, optimized_params)})
    return optimized_qc, mse_values, psnr_values, tv_values, combined_values, times


def save_images_as_tiff(images, base_filename):
    for i, image in enumerate(images):
        filename = f"{base_filename}_{i+1}.tif"
        imageio.imwrite(filename, image.astype(np.uint8), format='TIFF')
        print(f"Immagine salvata come {filename}")


def main(image_path, alpha=1, beta=1, gamma=1):
    # Preprocessa le immagini e calcola la GP
    print("Caricamento e preprocessing delle immagini...")
    red_channel, green_channel, gp_image_classic = load_and_preprocess_classical_images(image_path)
    
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
        n, m, gp_image_classic, angles_list, alpha, beta, gamma)


    print("Simulazione del circuito quantistico ottimizzato...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n+3))
    result = job.result()

    print("Recupero dei risultati...")
    counts = result.get_counts()
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)

    # Visualizza i risultati
    titles = [("Canale Verde", "Canale Rosso","Immagine GP Classica"), "Immagine GP Quantistica"]
    plot_images_strip([green_channel, red_channel, gp_image_classic], quantum_images, titles=titles)

    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, "quantum_image_output")

    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff"(green_channel"," red_channel", "gp_image_classic", "quantum01","quantum02")

    print("Fine del programma.")


# Parametri del filtro Gaussian e mediano
sigma = 1
size = 3
input_directory = 'IMMAGINI PER QUANTUM/trainQML'  # Cambia questo percorso
image_files = glob(os.path.join(input_directory, '*.tif'))


if __name__ == "__main__":
    # Percorsi delle immagini RGB
    image_path = "IMMAGINI PER QUANTUM/trainQML/membraneStack_Sample011_L_UV_DC_001rbc0DM2.tif"
  
    
 
    
  
    main(image_path, alpha=0.001, beta=0.1, gamma=0.1)
    
    
    
"""
    red, green, gp_classical = load_and_preprocess_classical_images(image_paths)
    
    # Recupera le immagini quantistiche generate dal software
    quantum_images_paths = ["quantum_image_output_1.tif", "quantum_image_output_2.tif"]  # Aggiornare con i nomi corretti
    quantum_images = []
    for path in quantum_images_paths:
        try:
            img = np.array(Image.open(path), dtype=np.float32)
            print(f"Caricata immagine quantistica da {path} con range {img.min()} - {img.max()}")
            quantum_images.append(img)
        except Exception as e:
            print(f"Errore nel caricamento di {path}: {e}")
            quantum_images.append(None)
    
    titles = ["Canale Verde", "Canale Rosso", "Immagine GP Classica"]
    titles += [f"Immagine GP Quantistica {i+1}" for i in range(len(quantum_images))]
    
    #plot_images_strip([green, red, gp_classical], quantum_images, titles)
"""  