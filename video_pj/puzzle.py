import random
import threading
import cv2
import multiprocessing as mp
import numpy as np

# Crear una barrera para dos hilos
barrier = threading.Barrier(1)

# Variables equivalentes
cols, rows = 2, 2
w, h = 400 // cols, 400 // rows
tiles = []
board = []

class Tile:
    def __init__(self, index, img):
        self.index = index
        self.img = img

# Configuración inicial
def setup():
    global tiles, board, solved_board
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
    # Guarda una copia del tablero resuelto
    solved_board = board.copy()
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
    return board == solved_board

# Manejador de eventos de ratón
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        i = x // w
        j = y // h
        move(i, j, board)

setup()

def process_frame_concurrent(q_in, q_out, cap):
    puzzle_completed = False
    while True:
        frame = q_in.get()
        if frame is None:
            break
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
            if not puzzle_completed:
                cv2.putText(final_img, 'Completado!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Rompecabezas", final_img)
                cv2.waitKey(5000)  # Muestra el mensaje durante 2 segundos
                cv2.destroyWindow("Rompecabezas")  # Cierra la ventana
                puzzle_completed = True
        else:
            puzzle_completed = False

        q_out.put(frame)
    barrier.wait()
