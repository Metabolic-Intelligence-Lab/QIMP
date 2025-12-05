import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio  # Correzione per il deprecation warning

from qiskit import QuantumCircuit, transpile
from math import pi, log2, ceil, sqrt
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import RYGate, MCMT, HGate
from qiskit.quantum_info import Operator

# Importa tutte le funzioni definite nel tuo script aggiuntivo
from FQRI_lib import (image_to_angles, setup_quantum_circuit, measure_quantum_circuit, 
                      apply_coordinate_swapping, apply_amplitude_amplification, 
                      apply_color_inversion, rotate_90, rotate_180, rotate_270, 
                      apply_color_transformation, neutralize_color, apply_custom_color_transformation,
                      save_image_as_tiff, extract_index)

class QuantumImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.qc = None  # Inizializza il circuito quantistico
        self.n = None  # Inizializza il numero di qubit per dimensione
        self.theta_values = None  # Memorizza i valori di theta
        self.initUI()

    def initUI(self):
        # Layout
        self.layout = QVBoxLayout()

        # Load image button
        self.btnLoad = QPushButton('Load Image', self)
        self.btnLoad.clicked.connect(self.openFileNameDialog)

        # Transformation buttons
        self.btnRotate90 = QPushButton('Rotate 90°', self)
        self.btnRotate90.clicked.connect(self.rotate90)

        self.btnRotate180 = QPushButton('Rotate 180°', self)
        self.btnRotate180.clicked.connect(self.rotate180)

        self.btnRotate270 = QPushButton('Rotate 270°', self)
        self.btnRotate270.clicked.connect(self.rotate270)

        self.btnColorInversion = QPushButton('Invert Colors', self)
        self.btnColorInversion.clicked.connect(self.invertColors)

        self.btnColorTransformation = QPushButton('Color Transformation', self)
        self.btnColorTransformation.clicked.connect(self.colorTransformation)

        self.btnNeutralizeColor = QPushButton('Neutralize Color', self)
        self.btnNeutralizeColor.clicked.connect(self.neutralizeColor)

        # Quit button
        self.btnQuit = QPushButton('Quit', self)
        self.btnQuit.clicked.connect(QApplication.quit)

        # Display images
        self.imageLabel = QLabel(self)
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)

        # Layout setup
        self.layout.addWidget(self.btnLoad)

        # Add transformation buttons
        self.layout.addWidget(self.btnRotate90)
        self.layout.addWidget(self.btnRotate180)
        self.layout.addWidget(self.btnRotate270)
        self.layout.addWidget(self.btnColorInversion)
        self.layout.addWidget(self.btnColorTransformation)
        self.layout.addWidget(self.btnNeutralizeColor)
        self.layout.addWidget(self.btnQuit)

        self.layout.addWidget(self.imageLabel)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Quantum Imaging')
        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        try:
            fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;TIFF Files (*.tif)", options=options)
            if fileName:
                self.image_path = fileName
                self.displayImage(fileName)
        except Exception as e:
            print(f"Error loading file: {e}")

    def displayImage(self, filePath):
        try:
            pixmap = QPixmap(filePath)
            self.imageLabel.setPixmap(pixmap.scaled(256, 256))
            self.processQuantumImage(filePath)
        except Exception as e:
            print(f"Error displaying image: {e}")

    def processQuantumImage(self, image_path):
        try:
            # Load image and calculate number of qubits
            image = imageio.imread(image_path)
            num_pixels = image.shape[0] * image.shape[1]  # Assuming image is square
            n_qubits = ceil(log2(num_pixels))
            self.n = n_qubits // 2  # Since we use 2*n qubits

            # Load and process image
            self.theta_values, self.n = image_to_angles(image_path)
            self.image_classic = np.array(self.theta_values) * 255 / pi
            self.image_classic = self.image_classic.reshape(int(sqrt(num_pixels)), int(sqrt(num_pixels)))

            # Setup and run quantum circuit
            self.qc = setup_quantum_circuit(self.n, self.theta_values)

            # Initial measure to show the original state
            measure_quantum_circuit(self.qc)
            self.simulateAndPlot()
        except Exception as e:
            print(f"Error processing quantum image: {e}")

    def simulateAndPlot(self):
        try:
            # Setup the quantum simulator
            simulator = AerSimulator()
            transpiled_circuit = transpile(self.qc, simulator)
            result = simulator.run(transpiled_circuit, shots=10**(self.n+2)).result()
            counts = result.get_counts()

            # Displaying the histogram on the canvas
            self.ax[0].clear()
            self.ax[0].imshow(self.image_classic, cmap='gray', vmin=0, vmax=255)
            self.ax[0].set_title("Immagine Classica")

            total_shots = sum(counts.values())
            prob = {k: v / total_shots for k, v in counts.items()}
            amp = {k: sqrt(v) for k, v in prob.items()}

            # Quantum image decoding
            prob_cond_0 = {}
            prob_cond_1 = {}
            values_q = {}
            for state, p in prob.items():
                j = state[1:]  # Prendiamo i due bit più a destra come 'j' per la posizione, il qubit del colore è il più a sinistra
                if state[0] == '0':
                    prob_cond_0[j] = p
                else:
                    prob_cond_1[j] = p
            for j in set(prob_cond_0.keys()) | set(prob_cond_1.keys()):
                p_0 = prob_cond_0.get(j, 0)
                p_1 = prob_cond_1.get(j, 0)
                if p_0 + p_1 > 0:
                    values_q[j] = np.arccos(sqrt(p_0 / (p_0 + p_1))) * 255 * (2 / np.pi)
                else:
                    values_q[j] = 0  # Imposta theta a 0 se non ci sono probabilità

            num_digits = len(next(iter(values_q.keys())))
            sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: extract_index(item[0]))}
            image_quantum = np.zeros((int(sqrt(2**(2*self.n))), int(sqrt(2**(2*self.n)))))
            for index, value in sorted_pixel_values.items():
                row = int(index[:num_digits // 2], 2)
                col = int(index[num_digits // 2:], 2)
                image_quantum[row][col] = value

            self.ax[1].clear()
            self.ax[1].imshow(image_quantum, cmap='gray', vmin=0, vmax=255)
            self.ax[1].set_title("Immagine Quantistica Ricostruita")

            # Redraw the canvas to update the GUI
            self.canvas.draw_idle()

            # Save quantum image as TIFF
            save_image_as_tiff(image_quantum, "quantum_image_output.tif")

        except Exception as e:
            print(f"Error simulating quantum circuit: {e}")

    # Transformation functions
    def rotate90(self):
        try:
            self.qc = setup_quantum_circuit(self.n, self.theta_values)  # Reset circuit before applying transformation
            rotate_90(self.qc, self.n)
            measure_quantum_circuit(self.qc)
            self.simulateAndPlot()
        except Exception as e:
            print(f"Error rotating 90°: {e}")

    def rotate180(self):
        try:
            self.qc = setup_quantum_circuit(self.n, self.theta_values)  # Reset circuit before applying transformation
            rotate_180(self.qc, self.n)
            measure_quantum_circuit(self.qc)
            self.simulateAndPlot()
        except Exception as e:
            print(f"Error rotating 180°: {e}")

    def rotate270(self):
        try:
            self.qc = setup_quantum_circuit(self.n, self.theta_values)  # Reset circuit before applying transformation
            rotate_270(self.qc, self.n)
            measure_quantum_circuit(self.qc)
            self.simulateAndPlot()
        except Exception as e:
            print(f"Error rotating 270°: {e}")

    def invertColors(self):
        try:
            self.qc = setup_quantum_circuit(self.n, self.theta_values)  # Reset circuit before applying transformation
            apply_color_inversion(self.qc)
            measure_quantum_circuit(self.qc)
            self.simulateAndPlot()
        except Exception as e:
            print(f"Error inverting colors: {e}")

    def colorTransformation(self):
        try:
            self.qc = setup_quantum_circuit(self.n, self.theta_values)  # Reset circuit before applying transformation
            apply_color_transformation(self.qc)
            measure_quantum_circuit(self.qc)
            self.simulateAndPlot()
        except Exception as e:
            print(f"Error applying color transformation: {e}")

    def neutralizeColor(self):
        try:
            self.qc = setup_quantum_circuit(self.n, self.theta_values)  # Reset circuit before applying transformation
            neutralize_color(self.qc)
            measure_quantum_circuit(self.qc)
            self.simulateAndPlot()
        except Exception as e:
            print(f"Error neutralizing color: {e}")

def main():
    app = QApplication(sys.argv)
    ex = QuantumImageApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
