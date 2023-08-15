import cv2
import streamlit as st

from Human_pose import Human_Pose
st.title("Webcam Live Feed")
@st.cache_resource
def load_model():
    model=Human_Pose(model_path="./weights/checkpoint_iter_370000.pth")
    return model
col1, col2= st.columns(2)
with col1:
   run = st.checkbox('Run') 
with col2 :
    start_button=st.button("Start_model")

st.write("Load model complete")
col3,col4=st.columns(2)
with col3:
    FRAME_WINDOW = st.image([])
with col4:
    RESULTS=st.image([])
camera = cv2.VideoCapture(0)
model=load_model()
img=None
while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if start_button:
        img,_=model.infer(frame)
        RESULTS.image(img)
    FRAME_WINDOW.image(frame)

