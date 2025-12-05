

from FQRI_lib import *


QiskitRuntimeService.save_account(channel="ibm_quantum", token="a701629b4cb81190a4a3707b135362ca3dd24e69f34a6920f7973bd9376caace2c95491188c2699cfae1b0078ed6b30dcd0774238a2d40ad8dcc740d18930471", overwrite=True)
    
#Generazione dei valori di theta
n = 6
n_qubits = 2*n
n_shots = 10**(n+2)
num_pixels = 2**(n_qubits)
"""
center_size_ratio = 0.25  # Configura la dimensione del quadrato centrale
noise_intensity = 0.2  # Configura l'intensità dello spot noise
# Esempio di utilizzo
patterns = ['random', 'grid', 'circle', 'fractal']
pattern = 'grid'
"""

# Generazione dei valori di theta
#theta_values = initialize_state(n, pattern, 0.5)






# Esempio di utilizzo
image_name = "Clipboard-1.tif"  # Sostituisci "nome_immagine.tif" con il nome effettivo della tua immagine TIFF
theta_values = image_to_angles(image_name)
image_values = np.array(theta_values) * 255 /pi



#♣theta_values = theta_out
#Generazione del Circuito Quantistico per FRQI

qc = setup_quantum_circuit(n , theta_values)
#rotate_270(qc,n)
#apply_coordinate_swapping(qc, n)
#apply_x_on_coordinate(qc, n, 'x')
#apply_x_on_coordinate(qc, n, 'y')
#apply_color_inversion(qc)
#apply_color_transformation(qc)
#neutralize_color(qc)
#theta0 = np.pi/2  # Ad esempio, theta = pi/4
#apply_custom_color_transformation(qc, theta0)  # Applica la trasformazione al qubit 0
marked_states=[n_qubits]
#apply_coordinate_swapping(qc, n)
apply_amplitude_amplification(qc, marked_states)
#apply_coordinate_swapping(qc, n)

measure_quantum_circuit(qc)

print(qc)


#Esecuzione del Circuito Quantistico sul Simulatore
simulator = AerSimulator()
qc = transpile(qc, simulator)
result = simulator.run(qc, shots = n_shots).result()
counts = result.get_counts(qc)
plot_histogram(counts, title='FRQI')
total_shots = sum(counts.values())
prob = {k: v / total_shots for k, v in counts.items()}
amp = {k: np.sqrt(v) for k, v in prob.items()}

# Costruzione dello stato psi' come stringa per la visualizzazione
psi_prime = " |ψ'⟩ = "
psi_prime += " + ".join([f"{amp_val:.4f}|{state}⟩" for state, amp_val in amp.items()])
print(psi_prime)



# Calcolo delle probabilità condizionali P(j|0) e P(j|1)
prob_cond_0 = {}
prob_cond_1 = {}
values_q = {}

for state, p in prob.items():
    # Il qubit del colore è il più a sinistra, quindi guardiamo il primo bit
    j = state[1:]  # Prendiamo i due bit più a destra come 'j' per la posizione
    if state[0] == '0':
        prob_cond_0[j] = p
    else:
        prob_cond_1[j] = p

for j in set(prob_cond_0.keys()) | set(prob_cond_1.keys()):
    p_0 = prob_cond_0.get(j,0)
    p_1 = prob_cond_1.get(j, 0)
    if p_0 + p_1 > 0:
        values_q[j] = np.arccos(np.sqrt(p_0 / (p_0 + p_1))) * 255 * (2 / np.pi)
    else:
        values_q[j] = 0  # Imposta theta a 0 se non ci sono probabilità

# Stampa i valori calcolati
"""
print("Probabilità condizionali P(j|0):", prob_cond_0)
print("Probabilità condizionali P(j|1):", prob_cond_1)
print("Valori  retrieved:", values_q)
print("valori originali:", image_values)
"""
num_digits = len(next(iter(values_q.keys())))
sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: extract_index(item[0]))}
image_quantum = np.zeros((int(math.sqrt(num_pixels)),int( math.sqrt(num_pixels))))

for index, value in sorted_pixel_values.items():
    row = int(index[:num_digits // 2], 2)
    col = int(index[num_digits // 2:], 2)
    image_quantum[row][col] = value


# Reshape dei valori dei pixel in una matrice 2x2 per l'immagine
image_classic = image_values.reshape(int(math.sqrt(num_pixels)),int( math.sqrt(num_pixels)))

# Crea una figura e definisci due subplot affiancati
plt.figure(figsize=(10, 5))  # Dimensione della figura che contiene i subplot

# Subplot per l'immagine classica
plt.subplot(1, 2, 1)  # (righe, colonne, indice del subplot)
plt.imshow(image_classic, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Classica")

# Subplot per l'immagine quantistica ricostruita
plt.subplot(1, 2, 2)  # (righe, colonne, indice del subplot)
plt.imshow(image_quantum, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Quantistica Ricostruita")

plt.show()



# Supponendo che 'image_quantum' sia la matrice NxN che hai ottenuto dalla misurazione quantistica
def image_to_theta(image_quantum):
    # Normalizzazione dei valori a una scala da 0 a pi
    theta_out = (image_quantum / 255) * np.pi
    
    # Trasformazione della matrice in un vettore
    theta_out_vector = theta_out.flatten()
    
    return theta_out_vector

