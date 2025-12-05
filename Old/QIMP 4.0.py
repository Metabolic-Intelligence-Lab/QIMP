from FQRI_lib import *
#QiskitRuntimeService.save_account(channel="ibm_quantum", token="a701629b4cb81190a4a3707b135362ca3dd24e69f34a6920f7973bd9376caace2c95491188c2699cfae1b0078ed6b30dcd0774238a2d40ad8dcc740d18930471", overwrite=True)
from qiskit_aer import AerSimulator
from qiskit import transpile


def main():
    print("Caricamento immagine...")   
    image_name = "ratioch1_16.tif"
    theta_values, n = image_to_angles(image_name)    
    
    print("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit(n, theta_values)
    #qc = rotate_90(qc, n)
    measure_quantum_circuit(qc)
    qc.draw()
    print("Simulazione del circuito quantistico...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n + 1))
    result = job.result()
    qc.draw()
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)

    print("Decodifica dei risultati...")
    image_quantum = decode_quantum_image(counts, n)


    print("Visualizzazione delle immagini...")
    classic_image = np.array(theta_values) * 255 / np.pi
    classic_image = classic_image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n))))
    plot_images([classic_image], [image_quantum], titles=[("Immagine Classica", "Immagine Quantistica Ricostruita")])

    print("Salvataggio dell'immagine quantistica...")
    save_image_as_tiff(image_quantum, "quantum_image_output.tif")

if __name__ == "__main__":
    main()

    
