import csv
import os
import random
import cv2
import multiprocessing as mp
import threading
import queue
import numpy as np

# Crear una barrera para dos hilos
barrier = threading.Barrier(3)

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

# Variables equivalentes
cols, rows = 4, 4
w, h = 400 // cols, 400 // rows
tiles = []
board = []

class Tile:
    def __init__(self, index, img):
        self.index = index
        self.img = img

# Configuración inicial
def setup():
    global tiles, board
    for i in range(cols):
        for j in range(rows):
            x, y = i * w, j * h
            img = np.zeros((h, w, 3), dtype=np.uint8)
            index = i + j * cols
            board.append(index)
            tile = Tile(index, img)
            tiles.append(tile)
    tiles.pop()
    board.pop()
    board.append(-1)
    simple_shuffle(board)

# Actualizar los mosaicos
def update_tiles(cap):
    global tiles
    for i in range(cols):
        for j in range(rows):
            x, y = j * w, i * h
            index = i + j * cols
            if index < len(tiles):
                ret, frame = cap.read()
                if ret:
                    tiles[index].img = cv2.resize(frame[y:y+h, x:x+w], (w, h))

# Funciones auxiliares
def swap(i, j, arr):
    arr[i], arr[j] = arr[j], arr[i]

def random_move(arr):
    r1 = random.randint(0, cols - 1)
    r2 = random.randint(0, rows - 1)
    move(r1, r2, arr)

def simple_shuffle(arr):
    for _ in range(1000):
        random_move(arr)

def is_neighbor(i, j, x, y):
    return (i == x and abs(j - y) == 1) or (j == y and abs(i - x) == 1)

def find_blank():
    return board.index(-1)

def move(i, j, arr):
    blank = find_blank()
    blank_col = blank % cols
    blank_row = blank // rows
    if is_neighbor(i, j, blank_col, blank_row):
        swap(blank, i + j * cols, arr)

def is_solved():
    for i in range(len(board) - 1):
        if board[i] != i:
            return False
    return True

# Manejador de eventos de ratón
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        i = x // w
        j = y // h
        move(i, j, board)

setup()

def process_frame_concurrent(q_in, q_out, cap):
    while True:
        frame = q_in.get()
        if frame is None:
            break
        # Implementar aquí
        update_tiles(cap)
        final_img = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(cols):
            for j in range(rows):
                index = i + j * cols
                x, y = i * w, j * h
                if index < len(board) and board[index] > -1:
                    img = tiles[board[index]].img
                    final_img[y:y+h, x:x+w] = img
        frame = final_img
        if is_solved():
            print("¡Completado!")
        q_out.put(frame)
    barrier.wait()
    
def show_frame(q, window_name):
    cv2.namedWindow(window_name)
    if window_name == 'Concurrent':
        cv2.setMouseCallback(window_name, click_event)
    while True:
        frame = q.get()
        if frame is None:
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



import time

def main():
    cap = cv2.VideoCapture(0)

    # Crear colas para compartir frames entre procesos/hilos
    q_parallel_in = mp.Queue()
    q_parallel_out = mp.Queue()
    q_smile_in = mp.Queue()  
    q_smile_out = mp.Queue()  
    q_concurrent_in = queue.Queue()
    q_concurrent_out = queue.Queue()

    # Crear un proceso para mostrar los frames procesados en paralelo
    start_time_parallel = time.time()
    p_parallel = mp.Process(target=process_frame_parallel, args=(q_parallel_in, q_parallel_out))
    p_parallel.start()
    t_parallel_show = threading.Thread(target=show_frame, args=(q_parallel_out, 'Parallel'))
    t_parallel_show.start() 

    # Crear un proceso para mostrar los frames con las sonrisas detectadas
    start_time_smile = time.time()
    p_smile = mp.Process(target=process_frame_smile, args=(q_smile_in, q_smile_out))
    p_smile.start()
    t_smile_show = threading.Thread(target=show_frame, args=(q_smile_out, 'Smile'))  
    t_smile_show.start() 

    # Crear un hilo para mostrar los frames procesados de manera concurrente
    start_time_concurrent = time.time()
    t_concurrent = threading.Thread(target=process_frame_concurrent, args=(q_concurrent_in, q_concurrent_out, cap))
    t_concurrent.start()
    t_concurrent_show = threading.Thread(target=show_frame, args=(q_concurrent_out, 'Concurrent'))
    t_concurrent_show.start() 
    # Verificar si el archivo existe
    file_csv='times.csv'
    if os.path.exists(file_csv):
        # Si existe, lo borramos para crear uno nuevo
        os.remove(file_csv)
    with open(file_csv, 'w', newline='') as file:
        writer = csv.writer(file)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            q_parallel_in.put(frame.copy())
            q_smile_in.put(frame.copy())  
            q_concurrent_in.put(frame.copy())

            # Calcular los tiempos totales aquí...
            total_parallel_time = time.time() - start_time_parallel
            total_smile_time = time.time() - start_time_smile
            total_concurrent_time = time.time() - start_time_concurrent

            # Escribir los tiempos en el archivo CSV
            writer.writerow([total_parallel_time, total_smile_time, total_concurrent_time])

    q_parallel_in.put(None)
    q_smile_in.put(None)  
    q_concurrent_in.put(None)

    cap.release()
    cv2.destroyAllWindows()
    p_parallel.join()
    p_smile.join()  
    t_parallel_show.join()
    t_smile_show.join()  
    t_concurrent.join()
    t_concurrent_show.join()



if __name__ == '__main__':
    mp.freeze_support()  # Añadir esta línea para soporte de congelación
    main()