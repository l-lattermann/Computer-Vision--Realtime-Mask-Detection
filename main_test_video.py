from ultralytics import YOLO
import cv2 
import time
import imageio

from scripts import cam_calibration as cc
from scripts import cv2_functions as cv2f

from config import YOLOV11_MASK, YOLOV8_MASK, VIDEO_DATA_DIR


def show_off(video_path):
    # Load models
    v8n_mask = YOLO(str(YOLOV8_MASK))  # YOLOv8n mask model
    v11n_mask = YOLO(str(YOLOV11_MASK))  # YOLOv11n mask model

    # Create model list
    model_dict = {"YOLOv8n":v8n_mask, "YOLOv11n": v11n_mask}

    # Initial values
    frame_count = 0 # Frame count
    stats_dict = {} # Stats dictionary
    avg_mask_size = 18 # Define average mask size   

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Save the video
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS

    # Add model variables to the stats dictionary
    stats_dict = {"Conf.": 0.7, "IOU": 0.5, "Inf. FPS": 1, "f": 100,"Model": "YOLOv8n", "Dist. test": False, "Blur": False}   # Initial values

    while True:
        # Fetch the latest frame
        ret, frame = cap.read()
        if not ret: # If the frame is not available,
            break   # Break the loop
        frame_count += 1  # Increment frame count

        # Reset frame count and start time every 100 frames
        if frame_count == 100:
            frame_count = 1

        # Define the model
        model = model_dict["YOLOv11n"]  # Get the model

        # Check every pred_framerate frames
        if frame_count % stats_dict["Inf. FPS"] == 0:
            # Perform inference for person and mask models
            inf_time = time.time()    # Start time 
            result = model(frame, conf=stats_dict["Conf."], iou=stats_dict["IOU"], stream=True, verbose=False)
            result = list(result)   # Convert generator to list
            stats_dict["Inference time"] = (time.time() - inf_time)*1000    # End time

        # Add bounding boxes to the frame
        box_time = time.time()    # Start time
        cv2f.put_bounding_boxes(frame, result, model, color_static=False)
        stats_dict["Annotation time"]=(time.time() - box_time)*1000    # End time

        # Add distance line to the frame
        line_time = time.time()    # Start time
        cv2f.put_distance_line(frame=frame, result=result, stats_dict=stats_dict, distance_threshold=100, avg_mask_size=avg_mask_size, font_scale=0.4, all_values=False)
        stats_dict["Line time"]=(time.time() - line_time)*1000    # End time
        
        # Key commands
        if not cv2f.wait_for_key(stats_dict, model_dict):  # Wait for key press
            break

        # Blur the faces
        if stats_dict["Blur"]:
            # Check if faces are detected
            if len(result[0].boxes.cls) > 0:
                # Save last face coordinates
                last_face_cords = result
                cv2f.blur_face(frame, result)
            # If no faces are detected, blur at last face coordinates
            elif len(result[0].boxes.cls) == 0 and "last_face_cords" in locals():
                cv2f.blur_face(frame, last_face_cords)
        
        # Resize the frame to uniform size
        cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        # Display the frame
        cv2.imshow("Detections", frame)

        #Convert to RGB and save the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frames.append(frame)


    # Release the video capture and output video
    cap.release()
    
    # Save video as GIF
    name = video_path.stem
    imageio.mimsave(f"tests/video_test_output/{name}.gif", frames, fps=fps)

    cv2.destroyAllWindows() # Close all OpenCV windows

def main():
    # Define the video path
    video_path1 = VIDEO_DATA_DIR / "istockphoto-1217776095-640_adpp_is.mp4"
    video_path2 = VIDEO_DATA_DIR / "istockphoto-1267958948-640_adpp_is.mp4"
    video_path3 = VIDEO_DATA_DIR / "istockphoto-1212372184-640_adpp_is.mp4"

    show_off(video_path1)
    show_off(video_path2)
    show_off(video_path3)

if __name__ == "__main__":
    main()