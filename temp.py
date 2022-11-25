""" 
def predictVideo(video_file):
    imgpath = os.path.join('data/uploads', video_file.name)
    outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

    with open(imgpath, mode='wb') as f:
        f.write(video_file.read())  # save video to disk

    st_video = open(imgpath, 'rb')
    video_bytes = st_video.read()
    st.video(video_bytes)
    st.write("Uploaded Video")
    #run(weights='S.pt', source=imgpath, device=0) if device == 'cuda' else 
    run(weights='N.pt', source=imgpath, device='cpu')
    st_video2 = open(outputpath, 'rb')
    video_bytes2 = st_video2.read()
    st.video(video_bytes2)
    st.write("Model Prediction")
"""


"""     model = torch.hub.load('yolov5', 'custom', path='S.pt', force_reload=True, source='local') 
    model.conf = confidence
    if len(classes):
        model.classes = classes
    pred = model(image_file, size=640)
    predJSON = json.loads(pred.pandas().xyxy[0].to_json(orient="records"))
    predSet = set()
    for item in predJSON:
        predSet.add(item['name'])
    pred.render()  # render bbox in image

    for im in pred.ims:
        im_base64 = Image.fromarray(im)
    return im_base64 """


#b4 adjust image funct
from cgitb import enable
from tkinter.tix import COLUMN
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import os  
import torch
#from detect import detect
from yolov5.detect import run
import json

modelForWebcam = torch.hub.load('yolov5', 'custom', path='M.pt', _verbose=False, source='local')
#! State------------------------------
if 'disabled_btn' not in st.session_state:
    st.session_state.disabled_btn = False
st.cache()
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
def load_image(img):
    im = Image.open(img)
    return im
def returnLang(thai, eng, lang):
    if(lang == 'ไทย'):
         return thai
    else:
        return eng
def predictImage(image_file, confidence, classes):
    run(
        weights='M.pt', 
        source=image_file, 
        device='cpu', 
        conf_thres=confidence,
        classes=classes
    ) 

def predictWebcam(frame):
    img = frame.to_ndarray(format="bgr24")
   # flipped = img[:, ::-1, :]
   # im_pil = Image.fromarray(flipped)
    results = modelForWebcam(img, size=640)
    bbox_img = np.array(results.render()[0])
    return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")

def disableBtn(uplType:str):
    if uplType == 'Video' or uplType == 'วิดีโอ':
        st.session_state.disabled_btn = not st.session_state.disabled_btn 
def main():
    st.markdown(
        """
            <style>
                #MainMenu {visibility: hidden;}
                .css-af4qln h3{color: #e16060; font-size:14px; padding:0;}
                .e16nr0p31 h2{padding-bottom:0;}
                label{color: #ff4b4b !important;}
                [data-testid="stSidebar"][aria-expanded="true"]{width:400px !important;}
                [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width:400px;}
                [data-testid="column"]{    
                    min-height: 100px;
                    border-radius: 8px;
                }
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child{background: #a4b6dd;}
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2){background: #d09292;}
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3){background: #c094cc;}
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4){background: #a2d0c0;}
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(5){background: #c37892;}
                [data-testid="stImage"]{margin:auto;}
            </style>  
        """,
        unsafe_allow_html=True
    )
    st.sidebar.title('ภาษาของโปรแกรม / Application Language')
    lang = st.sidebar.radio(
        label='',
        label_visibility='collapsed',
        options=('ไทย', 'English'),
        horizontal=True,
        disabled=st.session_state.disabled_btn
    )
    if lang == 'ไทย':
        diseaseName = ['โรคใบไหม้','โรคขอบใบแห้ง','โรคใบจุดสีน้ำตาล','โรคกาบใบแห้ง','โรคใบสีส้ม']
        inputOption = ('รูปภาพ', 'วิดีโอ', 'กล้องเว็บแคม')
    else:
        diseaseName = ['Rice Blast','Bacterial Blight','BrownSpot','Sheath Blight','Tungro']
        inputOption = ('Image', 'Video', 'Webcam Camera')
    st.title(returnLang(
            'โปรแกรมตรวจสอบโรคใบข้าวด้วย YOLOV5',
            'Rice leaf disease detection with YOLOV5',
            lang
        )
    )
    st.sidebar.markdown('---')
    st.sidebar.title(returnLang('ตั้งค่าโปรแกรม','Application Setting',lang))
#! Confidence------------------------------
    st.sidebar.header(returnLang('ค่าความเชื่อมั่น','Confidence',lang))
    confidence = st.sidebar.slider(returnLang(
        'เลือกค่าความเชื่อมั่นที่ต้องการ',
        'Select your specific confidence.',
        lang), 
        min_value = 0.0, max_value = 1.0, value = 0.25,
        disabled=st.session_state.disabled_btn    
    )
    modelForWebcam.conf = confidence
#! Checkbox------------------------------        
    custom_classes = st.sidebar.checkbox(returnLang('เลือกคลาสเฉพาะ','Use Specific Classes',lang))
    assigned_class_id = []
    if custom_classes:
        if(lang == 'ไทย'):
            assigned_class = st.sidebar.multiselect('เลือกโรคใบข้าวที่ต้องการให้ประมวลผล', list(diseaseName),default='โรคใบจุดสีน้ำตาล')
        else:
            assigned_class = st.sidebar.multiselect('Select the custom classes', list(diseaseName),default='BrownSpot')
        for each in assigned_class:
            assigned_class_id.append(diseaseName.index(each))
    st.sidebar.markdown('---')
#! Input Radio------------------------------
    st.sidebar.header(returnLang('ประเภทข้อมูลที่ต้องการประมวลผล','Input Type',lang))
    file_upload = st.sidebar.radio(
            label='',
            options=inputOption,
            index=0,
            label_visibility='collapsed',
            key="multi_select",
            disabled=st.session_state.disabled_btn
        )

#! Video------------------------------
    if file_upload == 'Video' or file_upload == 'วิดีโอ':
        video_file_buffer = st.sidebar.file_uploader(
            returnLang('เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบสีดำด้านล่าง',
            'Select browse files or drag and drop your files below.',lang), 
            type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
            disabled=st.session_state.disabled_btn
        )
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if not video_file_buffer:
            demo_video = 'testBS.mov'
            #vid = cv2.VideoCapture(demo_video)
            st_video = open(demo_video, 'rb')
            st.write(st_video.name)
            video_path = st_video.name
            use_demo_vid = True

            #imgpath = os.path.join('data/uploads', 'testBS.mov')
            #outputpath = os.path.join('data/video_output', os.path.basename(imgpath))
        else:
            tffile.write(video_file_buffer.read())
            st_video = open(tffile.name, 'rb')
            video_path = st_video.name
            use_demo_vid = False
        st.sidebar.video(st_video)
        if use_demo_vid:
            st.sidebar.warning(
                returnLang('คุณกำลังใช้วิดีโอตัวอย่างในการประมวลผล',
                    "You're using demo video for processing.",lang
                ), icon="⚠️"
            )
            
#! Image------------------------------
    elif file_upload == 'Image' or file_upload == 'รูปภาพ':
        image_file = st.sidebar.file_uploader(
            returnLang('เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบสีดำด้านล่าง',
            'Select browse files or drag and drop your files below.',lang), 
            type=['jpeg','jpg','png'],
        )
        tffile = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        if not image_file: 
            image_file = Image.open('testBS.jpg')
            image_file = image_file.resize((640,640))
            use_demo_img = True
        else: 
            temp = image_file
            image_file = Image.open(temp)
            image_file = image_file.resize((640,640))
            use_demo_img = False
        st.write(image_file)
        st.sidebar.header(returnLang('รูปแบบของรูปภาพ','Enhancing Image',lang))
        enhance_type = st.sidebar.radio('', 
            ['Original','Contrast','Brightness'],
            label_visibility='collapsed'
        )
        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast',0.5,3.5, value=1.0)
            st.sidebar.error(returnLang('** การปรับค่ารูปแบบของรูปภาพอาจส่งผลต่อความแม่นยำในการประมวลผลได้',
                '** Enhancing image can effect processing accuracy.',
                lang
            ))
            enhancer = ImageEnhance.Contrast(image_file)
            image_file = enhancer.enhance(c_rate)
        if use_demo_img:
            st.sidebar.warning(
                returnLang('คุณกำลังใช้ภาพตัวอย่างในการประมวลผล',
                    "You're using demo image for processing.",lang
                ), icon="⚠️"
            )
        st.sidebar.image(image_file)

#! Webcam------------------------------
    elif file_upload == 'Webcam Camera' or file_upload == 'กล้องเว็บแคม':
        #webrtc_streamer(key="example")
        #webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
        webrtc_ctx = webrtc_streamer(
            key="active webcam",
            mode=WebRtcMode.SENDRECV,
            #rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=predictWebcam,
            media_stream_constraints={"video": True, "audio": False},
           # async_processing=False,
        )

    st.sidebar.markdown('---')
#! Process------------------------------
    start_btn = st.sidebar.button(
        returnLang('เริ่มต้นการประมวลผล','Start Processing',lang), 
        disabled=st.session_state.disabled_btn,
        key='process_btn',
        on_click=disableBtn,
        args=(file_upload, )
    )
    if start_btn:
        col1, col2, col3, col4, col5 = st.columns(5)
        with st.spinner(returnLang('กำลังประมวลผลวิดีโอ...','Processing Video...',lang)):
            if file_upload == 'Image' or file_upload == 'รูปภาพ':
                st.image(predictImage(image_file, confidence, assigned_class_id))
            if file_upload == 'Video' or file_upload == 'วิดีโอ':
                showDetectedVideo = st.empty()
                showClassDetected = st.empty()
                stop = st.button(
                    'Stop Processing Video...',
                    key='stop_btn',
                    on_click = disableBtn,
                    args = (file_upload, ),
                    type = 'primary'
                )
                if(stop): 
                    st.error('Stop processing !')
                    st.session_state['disabled_btn '] = True
                    st.experimental_rerun()
                run(
                    weights='N.pt', 
                    source=video_path, 
                    device='cpu', 
                    showDetectedVideo=showDetectedVideo,
                    showClassDetected=showClassDetected
                ) 
                st.session_state['disabled_btn '] = True
                #st.session_state.disabled_btn = True
            col1.write('Col1')
            col2.write('Col2')
            col3.write('Col3')
            col4.write('Col4')
            col5.write('Col5')
    else:
        if file_upload == 'Image' or file_upload == 'รูปภาพ':
            st.image(image_file)
        if file_upload == 'Video' or file_upload == 'วิดีโอ':
            st.video(st_video)
if __name__ == '__main__':
    try: 
        main()
    except SystemExit:
        pass

# https://www.youtube.com/watch?v=mxRH275SyAU 17.40



backgroudn COLUMN
//#a4b6dd
#d09292;
#c094cc
#a2d0c0



            #! Show Class--------------------------------
            if detectedClass:
                #showDetected.header(detectedClass)
                with detectedClassList[0]:
                    st.header('Rice Blast')
                with detectedClassList[1]:
                    st.header('Bacterial Blight')
                with detectedClassList[2]:
                    st.header('Brown Spot')
                with detectedClassList[3]:
                    st.header('Sheath Blight')
                with detectedClassList[4]:
                    st.header('Tungro')
                    st.header(detectedClass['Tungro'])



                                        detectedCol = st.container()
                    with detectedCol:
                        blst, bb = st.columns(2)
                        if detectedClass:
                            with blst:
                                blst.header('hello world')
                    #showDetected.image()



            with st.expander('Detected Class', True):
                if showDetected != None: 
                    with showDetected:
                        st.image(im0, channels="BGR", use_column_width=True)
                    with st.expander('BrownSpot', True):
                        st.header('hello world')





 


 #==========B4 session video

 from cgitb import enable
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import os  
import torch
#from detect import detect
from yolov5.detect import run
import json
#! Check----------------------
'''
    - ดูภาษาตอน Process     
'''

#!----------------------
modelForWebcam = torch.hub.load('yolov5', 'custom', path='M.pt', _verbose=False, source='local')
#! State------------------------------
if 'disabled_btn' not in st.session_state:
    st.session_state.disabled_btn = False

if 'break_video' not in st.session_state:
    st.session_state.break_video = False

st.cache()
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
def load_image(img):
    im = Image.open(img)
    return im
def returnLang(thai, eng, lang):
    if(lang == 'ไทย'):
         return thai
    else:
        return eng
def predictImage(image_file, confidence, classes):
    run(
        weights='M.pt', 
        source=image_file, 
        device='cpu', 
        conf_thres=confidence,
        classes=classes,
        imgsz=(640, 640)
    ) 

def predictWebcam(frame):
    img = frame.to_ndarray(format="bgr24")
   # flipped = img[:, ::-1, :]
   # im_pil = Image.fromarray(flipped)
    results = modelForWebcam(img, size=640)
    bbox_img = np.array(results.render()[0])
    return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")

def disableBtn(uplType:str):
    if uplType == 'Video' or uplType == 'วิดีโอ':
        st.session_state.disabled_btn = not st.session_state.disabled_btn 
        st.session_state.break_video = False

def breakVideo(uplType:str):
    if uplType == 'Video' or uplType == 'วิดีโอ':
        st.session_state.disabled_btn = not st.session_state.disabled_btn 
        st.session_state.break_video = not st.session_state.break_video
def main():
    st.markdown(
        """
            <style>
                #MainMenu {visibility: hidden;}
                .css-af4qln h3{color: #e16060; font-size:14px; padding:0;}
                .e16nr0p31 h2{padding-bottom:0;}
                .streamlit-expanderHeader{font-size:1.5rem;}
                label{color: #ff4b4b !important;}
                [data-testid="stSidebar"][aria-expanded="true"]{width:400px !important;}
                [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width:400px;}
                [data-testid="column"]{    
                    min-height: 100px;
                    border-radius: 8px;
                }
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child{background: #ff9897;} 
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2){background: #feb2ac;}
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3){background: #f6a576;}
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4){background: #f2c46b;}
                [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(5){background: #ced05d;}
                [data-testid="stImage"]{margin:auto;}
                .st-ic > div > div > div > p{font-size:1.8rem;}
            </style>  
        """,
        unsafe_allow_html=True
    )
    print('================================hellllllllo worldddddd======================')
    st.sidebar.title('ภาษาของโปรแกรม / Application Language')
    lang = st.sidebar.radio(
        label='',
        label_visibility='collapsed',
        options=('ไทย', 'English'),
        horizontal=True,
        disabled=st.session_state.disabled_btn
    )
    if lang == 'ไทย':
        diseaseName = ['โรคใบไหม้','โรคขอบใบแห้ง','โรคใบจุดสีน้ำตาล','โรคกาบใบแห้ง','โรคใบสีส้ม']
        inputOption = ('รูปภาพ', 'วิดีโอ', 'กล้องเว็บแคม')
    else:
        diseaseName = ['Rice Blast','Bacterial Blight','BrownSpot','Sheath Blight','Tungro']
        inputOption = ('Image', 'Video', 'Webcam Camera')
    st.title(returnLang(
            'โปรแกรมตรวจสอบโรคใบข้าวด้วย YOLOV5',
            'Rice leaf disease detection with YOLOV5',
            lang
        )
    )
    st.sidebar.markdown('---')
    st.sidebar.title(returnLang('ตั้งค่าโปรแกรม','Application Setting',lang))
#! Confidence------------------------------
    st.sidebar.header(returnLang('ค่าความเชื่อมั่น','Confidence',lang))
    confidence = st.sidebar.slider(returnLang(
        'เลือกค่าความเชื่อมั่นที่ต้องการ',
        'Select your specific confidence.',
        lang), 
        min_value = 0.0, max_value = 1.0, value = 0.25,
        disabled=st.session_state.disabled_btn    
    )
    modelForWebcam.conf = confidence
#! Checkbox------------------------------        
    custom_classes = st.sidebar.checkbox(returnLang('เลือกคลาสเฉพาะ','Use Specific Classes',lang))
    assigned_class_id = []
    if custom_classes:
        if(lang == 'ไทย'):
            assigned_class = st.sidebar.multiselect('เลือกโรคใบข้าวที่ต้องการให้ประมวลผล', list(diseaseName),default='โรคใบจุดสีน้ำตาล')
        else:
            assigned_class = st.sidebar.multiselect('Select the custom classes', list(diseaseName),default='BrownSpot')
        for each in assigned_class:
            assigned_class_id.append(diseaseName.index(each))
    st.sidebar.markdown('---')
#! Input Radio------------------------------
    st.sidebar.header(returnLang('ประเภทข้อมูลที่ต้องการประมวลผล','Input Type',lang))
    file_upload = st.sidebar.radio(
            label='',
            options=inputOption,
            index=0,
            label_visibility='collapsed',
            key="multi_select",
            disabled=st.session_state.disabled_btn
        )

#! Video------------------------------
    if file_upload == 'Video' or file_upload == 'วิดีโอ':
        process_type = 'video'
        video_file_buffer = st.sidebar.file_uploader(
            returnLang('เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบสีดำด้านล่าง',
            'Select browse files or drag and drop your files below.',lang), 
            type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
            disabled=st.session_state.disabled_btn
        )
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if not video_file_buffer:
            demo_video = 'testBS.mov'
            #vid = cv2.VideoCapture(demo_video)
            st_video = open(demo_video, 'rb')
            st.write(st_video.name)
            video_path = st_video.name
            use_demo_vid = True

            #imgpath = os.path.join('data/uploads', 'testBS.mov')
            #outputpath = os.path.join('data/video_output', os.path.basename(imgpath))
        else:
            tffile.write(video_file_buffer.read())
            st_video = open(tffile.name, 'rb')
            video_path = st_video.name
            use_demo_vid = False
        st.sidebar.video(st_video)
        if use_demo_vid:
            st.sidebar.warning(
                returnLang('คุณกำลังใช้วิดีโอตัวอย่างในการประมวลผล',
                    "You're using demo video for processing.",lang
                ), icon="⚠️"
            )
            
#! Image------------------------------
    elif file_upload == 'Image' or file_upload == 'รูปภาพ':
        process_type = 'image'
        image_file = st.sidebar.file_uploader(
            returnLang('เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบสีดำด้านล่าง',
            'Select browse files or drag and drop your files below.',lang), 
            type=['jpeg','jpg','png'],
        )
        tffile = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        if not image_file: 
            image_file = Image.open('testBS.jpg')
            image_file_name = image_file.filename
            use_demo_img = True
        else: 
            tffile.write(image_file.read())
            temp_img = open(tffile.name, 'rb')
            image_file_name = temp_img.name
            use_demo_img = False
        st.sidebar.header(returnLang('รูปแบบของรูปภาพ','Enhancing Image',lang))
        enhance_type = st.sidebar.radio('', 
            ['Original','Contrast','Brightness'],
            label_visibility='collapsed'
        )
        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast',0.5,3.5, value=1.0)
            st.sidebar.error(returnLang('** การปรับค่ารูปแบบของรูปภาพอาจส่งผลต่อความแม่นยำในการประมวลผลได้',
                '** Enhancing image can effect processing accuracy.',
                lang
            ))
            enhancer = ImageEnhance.Contrast(image_file)
            image_file = enhancer.enhance(c_rate)
        if use_demo_img:
            st.sidebar.warning(
                returnLang('คุณกำลังใช้ภาพตัวอย่างในการประมวลผล',
                    "You're using demo image for processing.",lang
                ), icon="⚠️"
            )
        st.sidebar.image(image_file)

#! Webcam------------------------------
    elif file_upload == 'Webcam Camera' or file_upload == 'กล้องเว็บแคม':
        process_type = 'webcam'
        #webrtc_streamer(key="example")
        #webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
        webrtc_ctx = webrtc_streamer(
            key="active webcam",
            mode=WebRtcMode.SENDRECV,
            #rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=predictWebcam,
            media_stream_constraints={"video": True, "audio": False},
           # async_processing=False,
        )

    st.sidebar.markdown('---')
#! Process------------------------------
    start_btn = st.sidebar.button(
        returnLang('เริ่มต้นการประมวลผล','Start Processing',lang), 
        disabled=st.session_state.disabled_btn,
        key='process_btn',
        on_click=disableBtn,
        args=(file_upload, )
    )
                
    st.write(st.session_state.break_video)
    if start_btn:
        with st.spinner(returnLang(f'กำลังประมวลผล{process_type}...',f'Processing {process_type}...',lang)):
            #! Image------------------------------
            if file_upload == 'Image' or file_upload == 'รูปภาพ':
                if len(assigned_class_id) > 0:
                    run(
                        weights='M.pt', 
                        source=image_file_name, 
                        device='cpu', 
                        classes=assigned_class_id,
                        conf_thres=confidence,
                    ) 
                else:
                    run(
                        weights='M.pt', 
                        source=image_file_name, 
                        device='cpu', 
                        conf_thres=confidence,
                    ) 
            #! Video------------------------------
            if file_upload == 'Video' or file_upload == 'วิดีโอ':
                stop = st.button(
                    'Stop Processing Video...',
                    key='stop_btn',
                    on_click = breakVideo,
                    args = (file_upload, ),
                    type = 'primary'
                )
                #if(stop): 
                #    st.error('Stop processing !')
                #    st.session_state['disabled_btn '] = True
                #    st.experimental_rerun()
                run(
                    weights='N.pt', 
                    source=video_path, 
                    device='cpu', 
                    breakVideo=st.session_state.break_video
                ) 
                st.session_state['disabled_btn '] = True
                #st.session_state.disabled_btn = True
    else:
        if file_upload == 'Image' or file_upload == 'รูปภาพ':
            st.image(image_file)
        if file_upload == 'Video' or file_upload == 'วิดีโอ':
            st.video(st_video)
if __name__ == '__main__':
    try: 
        main()
    except SystemExit:
        pass

# https://www.youtube.com/watch?v=mxRH275SyAU 17.40
