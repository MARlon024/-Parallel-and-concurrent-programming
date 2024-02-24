import csv
import os
import time
import cv2
import multiprocessing as mp
import threading
import queue
from puzzle import click_event,process_frame_concurrent
from fractal import process_frame_parallel,process_frame_smile


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



def main():
    cap = cv2.VideoCapture(0)

    # Create queues to share frames between processes/threads
    q_parallel_in = mp.Queue()
    q_parallel_out = mp.Queue()
    q_smile_in = mp.Queue()  
    q_smile_out = mp.Queue()  
    q_concurrent_in = queue.Queue()
    q_concurrent_out = queue.Queue()

    # Create a process to display the frames processed in parallel
    start_time_parallel = time.time()
    p_parallel = mp.Process(target=process_frame_parallel, args=(q_parallel_in, q_parallel_out))
    p_parallel.start()
    t_parallel_show = threading.Thread(target=show_frame, args=(q_parallel_out, 'Parallel'))
    t_parallel_show.start() 

    # Create a process to display the frames with the detected smiles
    start_time_smile = time.time()
    p_smile = mp.Process(target=process_frame_smile, args=(q_smile_in, q_smile_out))
    p_smile.start()
    t_smile_show = threading.Thread(target=show_frame, args=(q_smile_out, 'Smile'))  
    t_smile_show.start() 

    # Create a thread to display the frames processed concurrently
    start_time_concurrent = time.time()
    t_concurrent = threading.Thread(target=process_frame_concurrent, args=(q_concurrent_in, q_concurrent_out, cap))
    t_concurrent.start()

    t_concurrent_show = threading.Thread(target=show_frame, args=(q_concurrent_out, 'Concurrent'))
    t_concurrent_show.start() 
    # Check if the file exists
    file_csv='times.csv'
    if os.path.exists(file_csv):
        # If it exists, we delete it to create a new one
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

            # Calculate total times here...
            total_parallel_time = time.time() - start_time_parallel
            total_smile_time = time.time() - start_time_smile
            total_concurrent_time = time.time() - start_time_concurrent

            # Write the times in the CSV file
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
    mp.freeze_support()  # Add this line for freeze support
    main()