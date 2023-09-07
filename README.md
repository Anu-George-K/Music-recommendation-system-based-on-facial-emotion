# Music-recommendation-system-based-on-facial-emotion

import cv2
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np
from deepface import DeepFace
from cv2 import *
import base64
from PIL import Image
from streamlit_player import st_player



def image_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
image_local('/home/anu/Documents/bigdata_nov/project1_facial emotion detection/images/62d07f662d83bf204cb213d9a456f210.jpg') 


def cartoonization (img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
    gray = cv2.medianBlur(gray, 9) 
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9) 
    color = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


st.title("MUSIC RECOMMENDATION SYSTEM BASED ON FACIAL EMOTION !!")
#run = st.checkbox('Run')
st.write("This is an app to play music based on your mood")

file_image = st.camera_input(label = "Take a pic of you to be sketched out and know your emotion")
font = cv2.FONT_HERSHEY_SIMPLEX
audio_emotion_map = {
    'sad': '/home/anu/Documents/bigdata_nov/streamlit_basics/sad-violin-150146.mp3',
    'neutral': '/home/anu/Documents/bigdata_nov/streamlit_basics/Pufino - Thoughtful (freetouse.com).mp3',
    'happy': '/home/anu/Documents/bigdata_nov/streamlit_basics/upbeat-happy-logo-2-versions-146604.mp3',
    'fear' : '/home/anu/Documents/bigdata_nov/streamlit_basics/cyber-war-126419.mp3',
    'surprise' :'/home/anu/Documents/bigdata_nov/streamlit_basics/leva-eternity-149473.mp3',
    'angry' : '/home/anu/Documents/bigdata_nov/streamlit_basics/cinematic-epic-trailer-clock-trailer-141369.mp3'

}

emoji_dict = {'sad': "https://images.rawpixel.com/image_png_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTEwL3JtNTg2LWZyb3duaW5nZmFjZS0wMV8xLWw5ZDNjMXYwLnBuZw.png",
    'neutral': "https://emojiisland.com/cdn/shop/products/Neutral_Emoji_icon_9f1cc93a-f984-4b6c-896e-d24a643e4c28_large.png?v=1571606091",
    'happy': "https://cdn.pixabay.com/photo/2020/12/27/20/24/smile-5865208_960_720.png",
    'fear' : "https://emojiisland.com/cdn/shop/products/Fearful_Face_Emoji_large.png?v=1571606037",
    'surprise' :"https://img.freepik.com/premium-vector/emoji-surprised-smiley-expression-emotes-like-vector-isolated-white-background_81894-5340.jpg?w=2000",
    'angry' : "https://w7.pngwing.com/pngs/666/250/png-transparent-angry-emoji-illustration-emoji-anger-emoticon-iphone-angry-emoji-orange-computer-wallpaper-smiley-thumbnail.png"
}
if file_image:
    input_img = Image.open(file_image)
    img_array = np.array(input_img)

    result_first = DeepFace.analyze(img_array,actions= ['emotion'],enforce_detection=False)
    result = {}
    for i in result_first:
        result.update(i)

    domi_emotion = result['dominant_emotion']
    final_sketch = cartoonization(img_array)
    cv2.putText(final_sketch, domi_emotion,(10,100),font,3,(0,0,255),3,cv2.LINE_4)

    st.write("** Pencil Sketch face with your emotion**")
    one = st.columns(1,gap="small")
    st.write('You are feeling...')
    
    st.image(final_sketch, use_column_width=True)
    for key,value in emoji_dict.items():
       if(key==domi_emotion):
           emoji_url = value  
           print(key)
           print(emoji_url) 
           col1, col2 = st.columns(2)
           with col1:
                st.title("You are feeling...")

           with col2:
                st.image(emoji_url, caption=key, width=100)
    
    
    st.write("#Music for u !!")
    #autoplay_audio(audio_emotion_map[domi_emotion])
    if(st.button("Play your recommended Music")):
            if(domi_emotion == 'happy'):
                st_player("https://www.youtube.com/watch?v=ZbZSe6N_BXs&list=PLplXQ2cg9B_qrNvF8KaDew3EetUqO8jBo")
            elif(domi_emotion == 'sad'):
                st_player("https://www.youtube.com/watch?v=ksM3UxHAWZA&list=PL5D7fjEEs5yflZzSZAhxfgQmN6C_6UJ1W&index=31")
            elif(domi_emotion == 'fear'):
                st_player("https://www.youtube.com/watch?v=XxkNj5hcy5E&list=PLt6CLmccrGjzIIKreE2M8-ZeWIuU-NE9d&index=5")
            elif(domi_emotion == 'angry'):
                st_player("https://www.youtube.com/watch?v=xqds0B_meys&list=PL_MH8gOS_ETiNT1NF8B46JYHZe6fXWfVW&index=4")
            elif(domi_emotion == 'surprise'):
                st_player("https://www.youtube.com/watch?v=rf1KGb1LTUI&list=PL2Hx2rn1E3W6WWPoNxTz3m_POvIJOBHOa&index=1")
   
            else:
                st_player("https://www.youtube.com/watch?v=tFGs7HP15d4")

       
       
       



else:
     st.write("You haven't uploaded any image file")



# run = st.button('click here to know your emotion')

FRAME_WINDOW = st.image([])
