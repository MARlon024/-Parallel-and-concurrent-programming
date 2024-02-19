import threading
import cv2
import multiprocessing as mp
import numpy as np

barrier = threading.Barrier(2)

# Cargar el archivo de cascada Haar
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_smile(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar sonrisas en la imagen
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20)

    return len(smiles) > 0  # Devolver True si se ha detectado al menos una sonrisa, False en caso contrario

def draw_smiles(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar sonrisas en la imagen
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20)

    # Dibujar un rectángulo alrededor de cada sonrisa detectada
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame  # Devolver el frame con las sonrisas detectadas

def process_frame_smile(q_in, q_out):
    while True:
        frame = q_in.get()
        if frame is None:
            break
        # Dibujar sonrisas
        smile_frame = draw_smiles(frame)
        q_out.put(smile_frame)  # Enviar el frame con las sonrisas detectadas a la cola de sonrisas
    barrier.wait()  # Mover esta línea fuera del bucle


def process_frame_parallel(q_in, q_out):
    last_frame = None
    iterations = 0
    smile_detected_counter = 0
    try:
        while True:
            frame = q_in.get()
            if frame is None:
                break
            # Detectar sonrisa
            smile_detected = detect_smile(frame)
            if smile_detected:
                smile_detected_counter += 1
                if smile_detected_counter == 100:
                    print("Fractal de Mandelbrot completado o generado")
                elif smile_detected_counter < 100:
                    print(f'Sonrisa detectada {smile_detected_counter} veces')  # Imprimir el número de detecciones de sonrisa
                iterations += 1
                
            last_frame = frame.copy()
            # Generar un fractal de Mandelbrot
            h, w = frame.shape[:2]
            y, x = np.ogrid[-1:1:h*1j, -2:1:w*1j]
            c = x + 1j*y
            z = c.copy()
            
            for _ in range(iterations):
                mask = np.abs(z) < 1000
                z[mask] = z[mask]**2 + c[mask]
            
            mandelbrot = (np.abs(z) < 1000)
            mandelbrot = mandelbrot.astype(np.uint8) * 255  # Convertir a una imagen en escala de grises
            mandelbrot = cv2.cvtColor(mandelbrot, cv2.COLOR_GRAY2BGR)  # Convertir de nuevo a BGR para que tenga 3 canales
            q_out.put(mandelbrot)
    except Exception as e:
        print("Ha ocurrido un error: ", str(e))
    finally:
        barrier.wait()  # Mover esta línea fuera del bucle