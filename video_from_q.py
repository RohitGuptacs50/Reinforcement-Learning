import cv2
import os
# Ceate video from q_table figure from q_learning_2 file 
def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

    for i in range(0, 4000, 100):
        img_path = f'/{i}.png'
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)
    
    out.release()

make_video()