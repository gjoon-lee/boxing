import cv2

def read_frames(path, flip=True):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():                
        raise FileNotFoundError(f"File Not Found{path}") 
    try:                                  
        while True:                      
            ret, frame = cap.read()
            if not ret:
                break
            if flip:                     
                frame = cv2.flip(frame,1)
            yield frame                  
    finally:                          
        cap.release()                   