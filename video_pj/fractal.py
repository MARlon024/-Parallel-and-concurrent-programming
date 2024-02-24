import threading
import cv2
import multiprocessing as mp
import numpy as np

barrier = threading.Barrier(2)
mutex = threading.Lock()  # Create a Mutex

# Load the Haar cascade file
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect smiles in the image
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20)

    return len(smiles) > 0  # Return True if at least one smile has been detected, False otherwise

def draw_smiles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect smiles in the image
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=20)

    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame  # Return the frame with the detected smiles

def process_frame_smile(q_in, q_out):
    while True:
        frame = q_in.get()
        if frame is None:
            break
        # Draw smiles
        smile_frame = draw_smiles(frame)
        q_out.put(smile_frame)  # Send the frame with the detected smiles to the smiles queue
    barrier.wait()  

def process_frame_parallel(q_in, q_out):
    last_frame = None
    iterations = 0
    smile_detected_counter = 0
    try:
        while True:
            frame = q_in.get()
            if frame is None:
                break
            # Detect smile
            smile_detected = detect_smile(frame)
            if smile_detected:
                with mutex:  # Use the Mutex to protect the shared data
                    smile_detected_counter += 1
                    if smile_detected_counter == 100:
                        print("Mandelbrot fractal completed or generated")
                    elif smile_detected_counter < 100:
                        print(f'Smile detected {smile_detected_counter} times')  # Print the number of smile detections
                    iterations += 1
                
            last_frame = frame.copy()
            # Generate a Mandelbrot fractal
            h, w = frame.shape[:2]
            y, x = np.ogrid[-1:1:h*1j, -2:1:w*1j]
            c = x + 1j*y
            z = c.copy()
            
            for _ in range(iterations):
                mask = np.abs(z) < 1000
                z[mask] = z[mask]**2 + c[mask]
            
            mandelbrot = (np.abs(z) < 1000)
            mandelbrot = mandelbrot.astype(np.uint8) * 255  # Convert to a grayscale image
            mandelbrot = cv2.cvtColor(mandelbrot, cv2.COLOR_GRAY2BGR)  # Convert back to BGR so it has 3 channels
            q_out.put(mandelbrot)
    except Exception as e:
        print("An error has occurred: ", str(e))
    finally:
        barrier.wait()
