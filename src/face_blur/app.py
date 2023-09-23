import streamlit as st
import io
import os
import cv2
import utils
from blur import FaceBlur

config_con = utils.read_config()
face_blur_con = config_con.get("face_blur")
artifact_con = config_con.get("artifact")

FACE_BLUR_DIR_NAME = os.path.join(artifact_con.get("root_dir_name"),face_blur_con.get("root_dir_name"))
face_blur = FaceBlur()

st.title("Face Detection")


uploaded_file = st.file_uploader("Choose a Video...", type=["mp4"])
temporary_location = False

if uploaded_file is not None:
    g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
    temporary_location = os.path.join(FACE_BLUR_DIR_NAME,face_blur_con.get("input_video_filename"))

    with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file

    ksize_value = int(st.slider("Select a Ksize Value", min_value=3, max_value=13, value=7))
    sigmax_value = float(st.slider("Select a SigmaX Value", min_value=0.5, max_value=100.0, value=25.0))
    if st.button("Detect Face"):
        face_blur.combine_all(ksize=ksize_value,sigmax=sigmax_value)
        print("<<<<<<   FACE DETECTION DONE   >>>>>>")

        output_video_path = os.path.join(FACE_BLUR_DIR_NAME,face_blur_con.get("output_video_filename"))
        
        with open(output_video_path,"rb") as f:
            con = f.read()
        st.download_button(label="Download The  Video",
            data=con,
            file_name='output_video.mp4',
        )