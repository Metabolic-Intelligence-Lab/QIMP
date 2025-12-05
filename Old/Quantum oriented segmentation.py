from FQRI_lib import *
#QiskitRuntimeService.save_account(channel="ibm_quantum", token="a701629b4cb81190a4a3707b135362ca3dd24e69f34a6920f7973bd9376caace2c95491188c2699cfae1b0078ed6b30dcd0774238a2d40ad8dcc740d18930471", overwrite=True)
from qiskit_aer import AerSimulator
from qiskit import transpile
from FQRI_lib import *



def main():
     
    print("Caricamento immagine...")   
    image_name = "Clipboard-1 32.tif"
    theta_values, n = image_to_angles(image_name)    
    n_qubits = 2 * n
    n_shots = 10**(n + 1)
    num_pixels = 2**(n_qubits)
    
    print("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit_single_image(n, theta_values)
    qc = neutralize_color(qc)
    measure_quantum_circuit(qc)

    print("Simulazione del circuito quantistico...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=n_shots)
    result = job.result()
    
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)

    print("Decodifica dei risultati...")
    image_quantum = decode_quantum_image(counts, n)

    print("Visualizzazione delle immagini...")
    plot_images_single(theta_values, image_quantum, n)

    print("Salvataggio dell'immagine quantistica...")
    save_image_as_tiff(image_quantum, "quantum_image_output.tif")

if __name__ == "__main__":
    main()
    


"""
image_name = "Clipboard-1 32.tif"
theta_values, n = image_to_angles(image_name)
n_qubits = 2 * n
n_shots = 10**(n + 1)
num_pixels = 2**(n_qubits)
image_values = np.array(theta_values) * 255 / np.pi
image_classic = image_values.reshape(int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels)))
qc = setup_quantum_circuit_single_image(n, theta_values)
#apply_color_inversion(qc)
#apply_color_inversion(qc)
#apply_amplitude_amplification(qc, marked_states)
#apply_color_inversion(qc)
measure_quantum_circuit(qc)


simulator = AerSimulator()
qc = transpile(qc, simulator)
result = simulator.run(qc, shots=n_shots).result()
counts = result.get_counts(qc)
plot_histogram(counts, title='FRQI')
total_shots = sum(counts.values())
prob = {k: v / total_shots for k, v in counts.items()}
amp = {k: np.sqrt(v) for k, v in prob.items()}
psi_prime = " |ψ'⟩ = " + " + ".join([f"{amp_val:.4f}|{state}⟩" for state, amp_val in amp.items()])


print("Decodifica dei risultati...")
image_quantum = decode_quantum_image(counts, n)
 
    
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_classic, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Classica")
plt.subplot(1, 2, 2)
plt.imshow(image_quantum, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Quantistica Ricostruita")
plt.show()

# Save quantum image as TIFF
save_image_as_tiff(image_quantum, "quantum_image_output.tif")
"""