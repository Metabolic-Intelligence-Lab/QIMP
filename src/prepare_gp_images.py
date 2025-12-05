import os
from glob import glob
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt

# Definizione della directory di input e output
input_directory = 'IMMAGINI PER QUANTUM/trainQML'  # Cambia questo percorso
output_directory = os.path.join(input_directory, 'GP_output')
train_qml_16_directory = os.path.join(input_directory, 'Train_QML_16')

# Crea la cartella di output se non esiste
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(train_qml_16_directory):
    os.makedirs(train_qml_16_directory)

# Parametri del filtro Gaussian e mediano
sigma = 1
size = 3

# Funzione per applicare filtri (senza soglia)
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

# Funzione per visualizzare i canali e l'immagine GP
def visualize_channels_and_gp(green_channel, red_channel, gp_image):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(green_channel, cmap='Greens')
    axs[0].set_title('Canale Verde')

    axs[1].imshow(red_channel, cmap='Reds')
    axs[1].set_title('Canale Rosso')

    axs[2].imshow(gp_image, cmap='gray')
    axs[2].set_title('Immagine GP')

    plt.show()

# Funzione per elaborare tutte le immagini nella directory
def process_images_in_directory(input_directory, output_directory):
    # Trova tutte le immagini RGB nella directory (ad esempio, immagini con estensione .tif)
    image_files = glob(os.path.join(input_directory, '*.tif'))
    
    if not image_files:
        print("Nessuna immagine trovata nella directory di input.")
        return
    
    # Cicla su ogni immagine trovata
    for image_file in image_files:
        try:
            image = Image.open(image_file)

            # Converti l'immagine in un array NumPy a 16-bit per gestire valori fino a 65535
            image_array = np.array(image, dtype=np.uint16)

            # Controlla che ci siano almeno 3 canali
            if image_array.ndim == 3 and image_array.shape[2] >= 3:
                red_channel = image_array[:, :, 0]
                green_channel = image_array[:, :, 1]

                # Applica i filtri (senza soglia)
                filtered_red_channel = apply_filters(red_channel)
                filtered_green_channel = apply_filters(green_channel)

                # Ridimensiona i canali a 16x16
                resized_red_channel = np.array(Image.fromarray(filtered_red_channel).convert('L').resize((16, 16), Image.LANCZOS))
                resized_green_channel = np.array(Image.fromarray(filtered_green_channel).convert('L').resize((16, 16), Image.LANCZOS))

                # Salva i canali ridimensionati
                red_output_filename = os.path.join(train_qml_16_directory, os.path.basename(image_file).replace('.tif', '_ch01_16x16.tif'))
                green_output_filename = os.path.join(train_qml_16_directory, os.path.basename(image_file).replace('.tif', '_ch02_16x16.tif'))
                Image.fromarray(resized_red_channel).save(red_output_filename)
                Image.fromarray(resized_green_channel).save(green_output_filename)

                # Calcola l'immagine GP
                gp_image = calculate_gp_image(resized_green_channel, resized_red_channel)  # Inverto red e green

                # Visualizza prima il canale verde, poi il rosso e infine l'immagine GP (disabilitato per esecuzione non-interattiva)
                # visualize_channels_and_gp(resized_green_channel, resized_red_channel, gp_image)

                # Salva l'immagine GP
                output_filename = os.path.join(output_directory, os.path.basename(image_file).replace('.tif', '_GP.tif'))
                Image.fromarray(gp_image).save(output_filename)

                print(f"Immagine GP salvata in {output_filename}")
            else:
                print(f"L'immagine {image_file} non contiene almeno 3 canali.")
        except OSError as e:
            print(f"Errore durante l'elaborazione dell'immagine {image_file}: {e}")

# Esegui l'elaborazione
process_images_in_directory(input_directory, output_directory)
