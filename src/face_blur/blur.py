import cv2
import cvzone
import supervision as sv
from utils import read_config
import os
import gdown
from ultralytics import YOLO

class FaceBlur:
    def __init__(self):
        self.config_content = read_config()
        self.root_dir_con = self.config_content['artifact']
        self.root_dir = self.root_dir_con['root_dir_name']
        os.makedirs(self.root_dir, exist_ok=True)

    def process(self, imarr,draw_rectangle=True):
        
        result = self.model.predict(imarr)
        for data in result[0].boxes.data:
            x1,y1,x2,y2 , _ , _ = data
            x1,y1,x2,y2  = int(x1),int(y1) ,  int(x2),int(y2)
            cv2.rectangle(imarr,(x1,y1) , (x2,y2) , (0,0,0)  )
            cutting_face = imarr[y1:y2,x1:x2]
            face_blur = cv2.GaussianBlur(cutting_face,(self.ksize,self.ksize),sigmaX = self.sigmax)
            imarr[y1:y2,x1:x2] = face_blur
        return imarr
    
    def download_model(self,model_dirve_id:str,model_path:str)->None:
        try:
            if not os.path.exists(model_path):
                gdown.download(id = model_dirve_id , output=model_path)
        except Exception as e:
            raise (e)
    def load_model(self,model_path):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise (e)
    def combine_all(self,ksize=7,sigmax=200.0):

        face_blur_con = self.config_content['face_blur']
        face_blur_root_dir = face_blur_con['root_dir_name']
        face_blur_dir_path = os.path.join(self.root_dir,face_blur_root_dir)
        self.ksize = ksize
        self.sigmax = sigmax
        os.makedirs(face_blur_dir_path,exist_ok=True)

        model_dirve_id = face_blur_con['model_dirve_id']
        model_path = os.path.join(face_blur_dir_path,face_blur_con['model_file_name'])
        self.download_model(model_dirve_id = model_dirve_id,model_path = model_path)
        self.load_model(model_path)
        input_video_path = os.path.join(face_blur_dir_path,face_blur_con['input_video_filename'])
        out_video_path = os.path.join(face_blur_dir_path,face_blur_con['output_video_filename'])

        sv.process_video(input_video_path,out_video_path,self.process)



if __name__ == '__main__':
    fb = FaceBlur()
    fb.combine_all(ksize=7,sigmax=200.0)