import cv2
import cvzone
import supervision as sv
from utils import read_config
import os
import gdown

class FaceBlur:
    def __init__(self):
        self.config_content = read_config()
        self.root_dir_con = self.config_content['artifact']
        self.root_dir = self.root_dir_con['root_dir_name']
        os.makedirs(self.root_dir, exist_ok=True)

    def process(self, imarr,draw_rectangle=False,thersold=20):
        
        faces = self.face_cascade.detectMultiScale(imarr,minSize=[150,150])
        for (x,y,w,h) in faces:
            x1,y1,x2,y2 = x-thersold,y-thersold,(x+w)+thersold,(y+h)+thersold
            if draw_rectangle:
                cv2.rectangle(imarr,(x1,y1),(x2,y2),(255,0,0),2)
            cutting_face = imarr[y1:y2,x1:x2]
            face_blur = cv2.GaussianBlur(cutting_face,(self.ksize,self.ksize),sigmaX = self.sigmax)
            imarr[y1:y2,x1:x2] = face_blur
        return imarr
    
    def download_cascade(self,cascade_id:str,cascade_path:str)->None:
        try:
            gdown.download(id = cascade_id , output=cascade_path)
        except Exception as e:
            raise (e)
    
    def combine_all(self,ksize,sigmax):
        self.ksize = ksize
        self.sigmax = sigmax
        face_blur_con = self.config_content['face_blur']
        face_blur_root_dir = face_blur_con['root_dir_name']
        face_blur_dir_path = os.path.join(self.root_dir,face_blur_root_dir)
        print(face_blur_dir_path)
        os.makedirs(face_blur_dir_path,exist_ok=True)

        cascade_id = face_blur_con['cascade_id']
        cascade_file_path = os.path.join(face_blur_dir_path,face_blur_con['cascade_file_name'])
        self.download_cascade(cascade_id = cascade_id,cascade_path = cascade_file_path)
        self.face_cascade = cv2.CascadeClassifier(cascade_file_path)

        input_video_path = os.path.join(face_blur_dir_path,face_blur_con['input_video_filename'])
        out_video_path = os.path.join(face_blur_dir_path,face_blur_con['output_video_filename'])

        sv.process_video(input_video_path,out_video_path,self.process)



if __name__ == '__main__':
    fb = FaceBlur()
    fb.combine_all(ksize=7,sigmax=200.0)