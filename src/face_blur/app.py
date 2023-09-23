# Import necessary libraries
import streamlit as st
import io
import os
import cv2
import utils
from blur import FaceBlur 

# Read configuration settings from a YAML file using the 'utils' module
config_con = utils.read_config()
face_blur_con = config_con.get("face_blur")
artifact_con = config_con.get("artifact")

# Define the directory name for face blurring results
FACE_BLUR_DIR_NAME = os.path.join(artifact_con.get("root_dir_name"), face_blur_con.get("root_dir_name"))

# Create an instance of the 'FaceBlur' class
face_blur = FaceBlur()

# Set the title of the Streamlit app
st.title("Face Detection")

# Create a file uploader widget for selecting a video file
uploaded_file = st.file_uploader("Choose a Video...", type=["mp4"])
temporary_location = False

# Check if a video file has been uploaded
if uploaded_file is not None:
    g = io.BytesIO(uploaded_file.read())  # Create a BytesIO object
    temporary_location = os.path.join(FACE_BLUR_DIR_NAME, face_blur_con.get("input_video_filename"))

    with open(temporary_location, 'wb') as out:
        out.write(g.read())  # Write the uploaded file to a temporary location as bytes

    # Create sliders for selecting blur parameters (ksize and sigmax)
    ksize_value = int(st.slider("Select a Ksize Value", min_value=3, max_value=13, value=7))
    sigmax_value = float(st.slider("Select a SigmaX Value", min_value=0.5, max_value=100.0, value=25.0))

    # Create a button to trigger face detection and blurring
    if st.button("Detect Face"):
        # Call the 'combine_all' method of the 'FaceBlur' instance
        face_blur.combine_all(ksize=ksize_value, sigmax=sigmax_value)
        print("<<<<<<   FACE DETECTION DONE   >>>>>>")

        # Get the path to the output video
        output_video_path = os.path.join(FACE_BLUR_DIR_NAME, face_blur_con.get("output_video_filename"))

        # Open the output video file and read its content
        with open(output_video_path, "rb") as f:
            con = f.read()

        # Create a download button for the output video
        st.download_button(label="Download The Video",
                           data=con,
                           file_name='output_video.mp4',
                           )
