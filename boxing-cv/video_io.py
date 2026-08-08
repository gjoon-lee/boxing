import cv2

def video_info(path):
    """Return container metadata as a dict: fps, width, height, frame_count.

    frame_count comes from the file header, not a real read loop
    trust it for browsing, reconcile with a loop when it matters.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {path}")
    try:
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()

def read_frames(path, flip=True):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():                
        raise FileNotFoundError(f"File Not Found {path}") 
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