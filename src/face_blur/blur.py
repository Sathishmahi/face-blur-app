# Import necessary libraries
import cv2
import cvzone
import supervision as sv
from utils import read_config
import os
import gdown
from ultralytics import YOLO

# Define a Python class called 'FaceBlur'
class FaceBlur:
    def __init__(self):
        # Read configuration data from a file
        self.config_content = read_config()
        self.root_dir_con = self.config_content['artifact']
        self.root_dir = self.root_dir_con['root_dir_name']
        # Create a directory specified in the configuration if it doesn't exist
        os.makedirs(self.root_dir, exist_ok=True)

    # Define a method 'process' for image processing with optional face detection
    def process(self, imarr, draw_rectangle=True):
        # Predict using a model (presumably loaded elsewhere)
        result = self.model.predict(imarr)
        # Loop through the predicted results
        for data in result[0].boxes.data:
            x1, y1, x2, y2, _, _ = data
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw a black rectangle around the detected face
            cv2.rectangle(imarr, (x1, y1), (x2, y2), (0, 0, 0))
            # Extract the detected face region
            cutting_face = imarr[y1:y2, x1:x2]
            # Apply Gaussian blur to the detected face
            face_blur = cv2.GaussianBlur(cutting_face, (self.ksize, self.ksize), sigmaX=self.sigmax)
            # Replace the detected face region with the blurred version
            imarr[y1:y2, x1:x2] = face_blur
        # Return the processed image
        return imarr

    # Define a method to download a model file from Google Drive
    def download_model(self, model_dirve_id: str, model_path: str) -> None:
        try:
            # Download the model if it doesn't exist
            if not os.path.exists(model_path):
                gdown.download(id=model_dirve_id, output=model_path)
        except Exception as e:
            raise (e)

    # Define a method to load a YOLO model from a given path
    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise (e)

    # Define a method to combine all steps, including model download and video processing
    def combine_all(self, ksize=7, sigmax=200.0):
        # Access configuration data related to face blurring
        face_blur_con = self.config_content['face_blur']
        face_blur_root_dir = face_blur_con['root_dir_name']
        face_blur_dir_path = os.path.join(self.root_dir, face_blur_root_dir)
        self.ksize = ksize
        self.sigmax = sigmax
        # Create a directory for face blurring if it doesn't exist
        os.makedirs(face_blur_dir_path, exist_ok=True)

        model_dirve_id = face_blur_con['model_dirve_id']
        model_path = os.path.join(face_blur_dir_path, face_blur_con['model_file_name'])
        # Download and load the YOLO model
        self.download_model(model_dirve_id=model_dirve_id, model_path=model_path)
        self.load_model(model_path)
        input_video_path = os.path.join(face_blur_dir_path, face_blur_con['input_video_filename'])
        out_video_path = os.path.join(face_blur_dir_path, face_blur_con['output_video_filename'])
        # Process the input video, applying face blurring
        sv.process_video(input_video_path, out_video_path, self.process)

# Entry point for the script
if __name__ == '__main__':
    # Create an instance of the 'FaceBlur' class
    fb = FaceBlur()
    # Call the 'combine_all' method to start the face blurring process with specified parameters
    fb.combine_all(ksize=7, sigmax=200.0)
