from ast import arg
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
from yolov5.detect import run
st.cache()
#! RTC Configuration----------------------
rtc_configuration = {
    'iceServers': [
        {'urls':['stun:stun.l.google.com:19302']}
    ]
}
modelForImage = torch.hub.load('yolov5', 'custom', path='M.pt', _verbose=False, source='local')
modelForWebcam = torch.hub.load('yolov5', 'custom', path='N.pt', _verbose=False, source='local')

detectedClass = []
detectedDict = {}
#! State------------------------------
if 'disabled_btn' not in st.session_state:
    st.session_state.disabled_btn = False

if 'break_video' not in st.session_state:
    st.session_state.break_video = False

if 'break_cam' not in st.session_state:
    st.session_state.break_cam = False

if 'allow_break_cam' not in st.session_state:
    st.session_state.allow_break_cam = False

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = ''


#! Page Setup----------------------
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    initial_sidebar_state= 'auto' 
)

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
    global detectedDict
    img = frame.to_rgb()
    img = img.to_ndarray() #Converet video frame to array
    results = modelForWebcam(img, size=(300,300))
    bbox_img = np.array(results.render()[0])
    return av.VideoFrame.from_ndarray(bbox_img)
    """ 
    json = results.pandas().xyxy[0].to_dict(orient='list')
    detectedClass = json['class']
    detectedDict = {i:detectedClass.count(i) for i in detectedClass} 
    bbox_img = np.array(results.render()[0])
    return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")"""

def disableBtn(uplType:str):
    if uplType == 'Video' or uplType == 'วิดีโอ' or uplType == 'กล้องเว็บแคม':
        st.session_state.disabled_btn = not st.session_state.disabled_btn 
        st.session_state.break_video = False

def breakVideo(uplType:str):
    if uplType == 'Video' or uplType == 'วิดีโอ':
        st.session_state.disabled_btn = not st.session_state.disabled_btn 
        st.session_state.break_video = not st.session_state.break_video

def breakCam(uplType:str):
    if uplType == 'กล้องเว็บแคม':
        st.session_state['disabled_btn'] = False
        st.session_state['break_cam'] = True
        st.session_state['allow_break_cam'] = True
def main():
    start_cam_btn = False
    st.markdown(
        """
            <style>
                //#MainMenu {visibility: hidden;}
                .option{
                    font-size:2rem !important;
                    border-bottom:1px solid #cecfd4;
                }
                .toggle{display:none;}
                .css-renyox{
                    top:10px;
                    right:10px;
                }
                .tab-subhead{
                    padding-top:0;
                    padding-bottom:5px;
                }
                .indent{text-indent:30px;}
                .bold{font-weight:bold;}
                .css-af4qln h3{color: #e16060; font-size:14px; padding:0;}
                .e16nr0p31 h2{padding-bottom:0;}
                .css-1valv9w > div > div > div > p{margin:0;}
                .streamlit-expanderHeader{font-size:1.8rem;}
                .h4-success,.h4-danger{
                    color: #71ab85;
                    background: #e9f9ef;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom:10px;
                }
                .h4-danger{
                    color: #d85078;
                    background: #f2e1e6;
                }
                .warning-webcam{
                    background:#fffcec;
                    color:#926c05;
                    text-align:center;
                }
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
                [data-testid="stMarkdownContainer"] ul{
                    list-style-position: inside;
                    list-style-type:none;
                }
                [data-baseweb="accordion"]{padding:.125em;}
                [data-testid="stCameraInput"] > label{
                    font-size:20px;
                    background:#978c54;
                    color:#efdd83 !important;
                    justify-content:center;
                }
                [data-testid="stCameraInput"] > div > button {
                    background:#cd5858;
                    color:white;
                    font-size:30px;
                }
                [data-stale="false"] > .row-widget > button {width: 100%; border-color:red;}
                .st-ic > div > div > div > p{font-size:1.8rem;}
                [data-testid="stVerticalBlock"] div:nth-child(3) > div > div:nth-child(1){}
                [data-testid="stVerticalBlock"] div:nth-child(3) > div > div:nth-child(2){width:100%;display:none;}
                [data-testid="collapsedControl"]{display: initial;}
                [data-testid="stAppViewContainer"] > section:first-child{display:initial;}
                [tabindex="0"] > div > div > div > div:nth-child(5), [tabindex="0"] > div > div > div > div:nth-child(6){
                    display:none;
                }
                @media only screen and (max-width: 425px) {
                    .toggle{
                        width:100%;
                        display:block;
                        text-decoration: none !important;
                        text-align: center;
                        border:1px solid red;
                        border-radius:5px;
                        color:black !important;
                    }
                    .invisible_btn{
                        position: absolute;
                        bottom: 20px;
                        width: 100%;
                        display: block;
                        border: 1px solid;
                        opacity: 0;
                    }
                    [data-testid="stAppViewContainer"] > section:first-child{width:300px !important;}
                    [data-testid="stAppViewContainer"] > section:first-child > div:first-child{ width:300px;}
                    [data-testid="stAppViewContainer"] > section:first-child > div:first-child > div:nth-child(2) > div > div > div > div:nth-last-child(2){
                        display:none;
                    }
                    [data-testid="stVerticalBlock"] div:nth-child(3) > div > div:nth-child(2){display:initial;}
                    section > .css-6qob1r .e1fqkh3o3{background: width:300px !important;}
                    [data-testid="stSidebar"] .block-container{max-width:250px;}
                    
                    [tabindex="0"] > div > div > div > div:nth-child(3) > div{
                    }
                    [tabindex="0"] > div > div > div > div:nth-child(5), [tabindex="0"] > div > div > div > div:nth-child(6){
                        display:initial;
                    }
                }
            </style>  
        """,
        unsafe_allow_html=True
    )
    diseaseName = ['โรคใบไหม้','โรคขอบใบแห้ง','โรคใบจุดสีน้ำตาล','โรคกาบใบแห้ง','โรคใบสีส้ม']
    inputOption = ('รูปภาพ', 'วิดีโอ', 'กล้องถ่ายรูป','กล้องเว็บแคม')
    st.title('โปรแกรมตรวจสอบโรคใบข้าวด้วย YOLOV5')
    st.markdown('''
     <a class="toggle" href="javascript:document.getElementsByClassName('css-9s5bis edgvbvh3')[1].click();" target="_self">ตั้งค่าโปรแกรม</a>
    ''', unsafe_allow_html=True)
    st.sidebar.markdown(f'<h4 class="option">ตั้งค่าโปรแกรม</h4>', unsafe_allow_html=True) 
#! Get Confidence------------------------------------------------------------------------------------------------------------------------
    st.sidebar.header('ค่าความเชื่อมั่น')
    confidence = st.sidebar.slider(
        'เลือกค่าความเชื่อมั่นที่ต้องการ',
        min_value = 0.0, max_value = 1.0, value = 0.25,
        disabled=st.session_state.disabled_btn    
    )
    modelForImage.conf = confidence
    modelForWebcam.conf = confidence
#! Get Specific Class------------------------------------------------------------------------------------------------------------------------        
    custom_classes = st.sidebar.checkbox('เลือกคลาสเฉพาะ')
    assigned_class_id = []
    if custom_classes:
        assigned_class = st.sidebar.multiselect('เลือกโรคใบข้าวที่ต้องการให้ประมวลผล', list(diseaseName),default='โรคใบจุดสีน้ำตาล')
        for each in assigned_class:
            assigned_class_id.append(diseaseName.index(each))
        if(len(assigned_class_id)):
            modelForImage.classes = assigned_class_id
    st.sidebar.markdown('---')

#! Get Input Type------------------------------------------------------------------------------------------------------------------------
    st.sidebar.header('ประเภทข้อมูลที่ต้องการประมวลผล')
    file_upload = st.sidebar.radio(
            label='',
            options=inputOption,
            index=0,
            label_visibility='collapsed',
            key="multi_select",
            disabled=st.session_state.disabled_btn
        )

#! Get Video------------------------------------------------------------------------------------------------------------------------
    if file_upload == 'วิดีโอ':
        process_type = 'วิดีโอ'
        video_file_buffer = st.sidebar.file_uploader(
            'เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบสีดำด้านล่าง', 
            type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
            disabled=st.session_state.disabled_btn
        )
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if not video_file_buffer:
            demo_video = 'testBS.mov'
            #vid = cv2.VideoCapture(demo_video)
            st_video = open(demo_video, 'rb')
            video_path = st_video.name
            use_demo_vid = True

            #imgpath = os.path.join('data/uploads', 'testBS.mov')
            #outputpath = os.path.join('data/video_output', os.path.basename(imgpath))
        else:
            tffile.write(video_file_buffer.read())
            st_video = open(tffile.name, 'rb')
            video_path = st_video.name
            use_demo_vid = False
        emptyVideo = st.empty()
        
        st.write(st_video.name)
        st.sidebar.video(st_video)
        emptyVideo.video(st_video)
        if use_demo_vid:
            st.sidebar.warning('คุณกำลังใช้วิดีโอตัวอย่างในการประมวลผล', icon="⚠️")
            
#! Get Image------------------------------------------------------------------------------------------------------------------------
    elif file_upload == 'Image' or file_upload == 'รูปภาพ':
        process_type = 'รูปภาพ'
        image_file = st.sidebar.file_uploader(
            'เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบสีดำด้านล่าง', 
            type=['jpeg','jpg','png'],
        )
        tffile = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        if not image_file: 
            image_file = Image.open('testBS.jpg')
            pilImg = image_file
            image_file = image_file.filename
            use_demo_img = True
        else: 
            tffile.write(image_file.read())
            temp_img = open(tffile.name, 'rb')
            pilImg = Image.open(temp_img)
            image_file = temp_img.name
            st.write('ประมวลผลไฟล์: ',image_file)
            use_demo_img = False
        st.sidebar.header('รูปแบบของรูปภาพ','Enhancing Image')
        enhance_type = st.sidebar.radio('', 
            ['Original','Contrast','Brightness'],
            label_visibility='collapsed'
        )
        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast',0.5,3.5, value=1.0)
            st.sidebar.error('** การปรับค่ารูปแบบของรูปภาพอาจส่งผลต่อความแม่นยำในการประมวลผลได้')
            enhancer = ImageEnhance.Contrast(pilImg)
            image_file = enhancer.enhance(c_rate)
        if enhance_type == 'Brightness':
            b_rate = st.sidebar.slider('Brightness',0.5,3.5, value=1.0)
            st.sidebar.error('** การปรับค่าความสว่างของรูปภาพอาจส่งผลต่อความแม่นยำในการประมวลผลได้')
            enhancer = ImageEnhance.Brightness(pilImg)
            image_file = enhancer.enhance(b_rate)
        if use_demo_img:
            st.sidebar.warning('คุณกำลังใช้ภาพตัวอย่างในการประมวลผล', icon="⚠️")
        st.sidebar.image(image_file)
        showImage = st.empty()
        showImage.image(image_file)

#! Get Webcam Image Capture------------------------------------------------------------------------------------------------------------------------
    elif file_upload == 'กล้องถ่ายรูป':
        camera = st.empty()
        img_capture = camera.camera_input("เลือก 'Take Photo' เพื่อถ่ายรูปที่ต้องการใช้สำหรับการประมวลผล", label_visibility='collapsed')
        if img_capture:
            process_type = 'กล้องถ่ายรูป'
            tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tffile.write(img_capture.read())
            temp_img = open(tffile.name, 'rb')
            pilImg = Image.open(temp_img)
            image_file = temp_img.name
            camera.image(img_capture)
            start_cam_btn = st.button(
                'เริ่มต้นการประมวลผล',
                key='start_cam_btn',
            )   
#! Get Webcam------------------------------------------------------------------------------------------------------------------------
    elif file_upload == 'Webcam Camera' or file_upload == 'กล้องเว็บแคม':
        translations={
            "start": "เริ่มต้นการประมวลผล",
            "stop": "หยุดการประมวลผล",
            "select_device": "เลือกอุปกรณ์",
        }
        webrtc_streamer(
            key="webcam_process", 
            video_frame_callback=predictWebcam,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            translations=translations,
            async_processing=True
        )
        """ process_type = 'กล้องเว็บแคม'
        camWarningText = st.empty()
        camWarningText.markdown(f'<h4 class="warning-webcam">เลือก "เริ่มต้นการประมวลผล" เพื่อใช้งานกล้องเว็บแคม</h4>', unsafe_allow_html=True) """
        #st.write(webrtc_ctx)
        #st.write(detectedDict)
        #webrtc_streamer(key="example")
        #webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
    st.sidebar.markdown('---')
#! Process Start------------------------------------------------------------------------------------------------------------------------
    #TODO Sidebar----
    start_btn = st.sidebar.button(
        'เริ่มต้นการประมวลผล', 
        disabled=st.session_state.disabled_btn,
        key='process_btn',
        on_click=disableBtn,
        args=(file_upload, )
    )
    
    if file_upload == 'รูปภาพ' or file_upload == 'วิดีโอ':
        st.write('-----')
        emptyStartBtn = st.empty()
        start_btn = emptyStartBtn.button(
            'เริ่มต้นการประมวลผล', 
            disabled=st.session_state.disabled_btn,
            key='process_btn1',
            on_click=disableBtn,
            args=(file_upload, )
        )

    st.sidebar.markdown('''
     <a class="toggle" href="javascript:document.getElementsByClassName('css-9s5bis edgvbvh3')[1].click();" target="_self">สิ้นสุดการตั้งค่าโปรแกรม</a>
    ''', unsafe_allow_html=True)


    if start_btn or (file_upload == 'กล้องถ่ายรูป' and start_cam_btn) or (st.session_state.break_cam and st.session_state.allow_break_cam):
        with st.spinner(f'กำลังประมวลผล{process_type}...'):
#! Process Image------------------------------------------------------------------------------------------------------------------------
            if file_upload == 'รูปภาพ' or file_upload == 'กล้องถ่ายรูป':
                modelForImage.conf = confidence
                if len(assigned_class_id) > 0:
                    modelForImage.classes = assigned_class_id
                    result = modelForImage(image_file, size=640)
                else:
                    result = modelForImage(image_file, size=640)
                if result:
                    json = result.pandas().xyxy[0].to_dict(orient='list')
                    detectedClass = json['class']
                    detectedDict = {i:detectedClass.count(i) for i in detectedClass}
                    r_img = result.render()
                    if file_upload == 'กล้องถ่ายรูป':
                        camera.image(r_img)
                    else: showImage.image(r_img)
#! Process Video------------------------------------------------------------------------------------------------------------------------
            elif file_upload == 'Video' or file_upload == 'วิดีโอ':
                stop = st.button(
                    'Stop Processing Video...',
                    key='stop_btn',
                    on_click = breakVideo,
                    args = (file_upload, ),
                    type = 'primary'
                )
                emptyVideo.empty()
                emptyStartBtn.empty()
                if len(assigned_class_id) > 0:
                    run(
                        weights='N.pt', 
                        source=video_path, 
                        device='cpu', 
                        breakVideo=st.session_state.break_video,
                        conf_thres=confidence,
                        classes=assigned_class_id
                    ) 
                else:
                    run(
                        weights='N.pt', 
                        source=video_path, 
                        device='cpu', 
                        breakVideo=st.session_state.break_video,
                        conf_thres=confidence,
                    ) 
                st.session_state['disabled_btn '] = False 

#! Process Webcam Camera------------------------------------------------------------------------------------------------------------------------
            elif file_upload == 'กล้องเว็บแคม':                                    
                if not st.session_state.allow_break_cam:
                    pass
                    """ camWarningText.empty()
                    stopWebcam = st.button(
                        'Stop Processing Webcam...',
                        key='stop_cam',
                        on_click = breakCam,
                        args = (file_upload, ),
                        type = 'primary'
                    )
                    if len(assigned_class_id) > 0:
                        run(weights='N.pt', 
                            source=0, 
                            device='cpu', 
                            breakVideo=st.session_state.break_cam,
                            conf_thres=confidence,
                            classes=assigned_class_id
                        ) 
                    else:
                        run(weights='N.pt',  
                            source=0, 
                            device='cpu', 
                            breakVideo=st.session_state.break_cam,
                            conf_thres=confidence,
                        ) 
                    st.session_state['disabled_btn'] = False
                    st.session_state['allow_break_cam'] = False
                    st.session_state['break_cam'] = False  """
               
#! Detected Result------------------------------------------------------------------------------------------------------------------------
        if (file_upload == 'รูปภาพ' or file_upload == 'กล้องถ่ายรูป') and (start_btn or start_cam_btn):
            with st.container():
                if detectedDict:
                    st.markdown('<h4 class="h4-success">ตรวจพบโรคใบข้าว</h4>', unsafe_allow_html=True)  
                    with st.empty():
                        with st.container():
                            #TODO Leaf Blast--------------------------------------------------------------------
                            if 0 in detectedDict:
                                with st.expander(f'ตรวจพบโรคใบไหม้ทั้งหมด: {detectedDict[0]} ตำแหน่ง',True):
                                    st.markdown(f'<h4 class="tab-subhead">โรคไหม้ (Rice Blast Disease)</h4>', unsafe_allow_html=True)
                                    st.markdown('<hr>',unsafe_allow_html=   True)  
                                    tab1,tab2,tab3,tab4,tab5 = st.tabs(['สาเหตุของโรคและภูมิภาคที่พบ','อาการของโรค','การแพร่ระบาด','การป้องกัน','ข้อควรระวัง'])
                                    with tab1:  
                                        st.markdown('<h5 class="tab-subhead">สาเหตุของโรค</h5> \
                                                    <p class="indent">เชื้อรา Pyricularia oryzae.</p> \
                                                    <h5 class="tab-subhead">ภูมิภาคที่พบ</h5> \
                                                    <p class="indent">พบส่วนใหญ่ในภาคเหนือ ภาคตะวันออกเฉียงเหนือ ภาคตะวันตก และภาคใต้ พบมากในข้าวนาสวน ทั้งนาปีและนาปรัง และข้าวไร่</p>'
                                                , unsafe_allow_html=True
                                        )  
                                    with tab2:
                                        st.markdown('<h5 class="tab-subhead">ระยะกล้า</h5> \
                                                <p class="indent">ใบมีแผลจุดสีน้ำตาลคล้ายรูปตามีสีเทาอยู่ตรงกลางแผลความกว้างของแผลประมาณ 2-5 มิลลิเมตร และความยาวประมาณ 10-15 มิลลิเมตร แผลสามารถขยายลุกลามและกระจายทั่วบริเวณใบถ้าโรครุนแรงกล้าข้าวจะแห้งฟุบตาย</p> \
                                                <h5 class="tab-subhead">ระยะแตกกอ</h5> \
                                                <p class="indent">อาการพบได้ที่ใบข้อต่อของใบ และข้อต่อของลำต้น ขนาดแผลจะใหญ่กว่าที่พบในระยะกล้าแผลลุกลามติดต่อกันได้ที่บริเวณข้อต่อ ใบจะมีลักษณะแผลช้ำสีน้ำตาลดำ และมักหลุดจากกาบใบเสมอ</p> \
                                                <h5>ระยะออกรวง</h5> \
                                                <p class="tab-subhead">ถ้าข้าวเพิ่งจะเริ่มให้รวงเมื่อถูกเชื้อราเข้าทำลายเมล็ดจะลีบหมด แต่ถ้าเป็นโรคตอนรวงข้าวแก่ใกล้เก็บเกี่ยวจะปรากฏรอยแผลช้ำสีน้ำตาลที่บริเวณคอรวง ำให้เปราะหักง่ายรวงข้าวร่วงหล่นเสียหายมาก</p> '
                                            ,unsafe_allow_html=True
                                        )
                                                            
                                    with tab3:
                                        st.markdown('<p class="indent">พบโรคในแปลงที่ต้นข้าวหนาแน่นทำให้อับลม ถ้าอากาศค่อนข้างเย็น อุณหถูมิประมาณ 22-25 องศาเซลเซียส ลมแรงจะช่วยให้โรคแพร่กระจายได้ดี</p>',unsafe_allow_html=True)
                                    with tab4:
                                        st.markdown('<p class="indent bold">ใช้พันธุ์ข้าวที่ค่อนข้างต้านทานโรค</p> \
                                                <p class="indent">- ภาคกลาง เช่น สุพรรณบุรี 1 สุพรรณบุรี 60 ปราจีนบุรี 1 พลายงาม ข้าวเจ้าหอมพิษณุโลก 1</p> \
                                                <p class="indent">- ภาคเหนือ และตะวันออกเฉียงเหนือ เช่น ข้าวเจ้าหอมพิษณุโลก 1 สุรินทร์ 1 เหนียวอุบล 2 สันปาตอง 1 หางยี 71 กู้เมืองหลวง ขาวโป่งไคร้ น้ำรู</p> \
                                                <p class="indent">- ภาคใต้ เช่น ดอกพะยอม</p> \
                                                <p class="indent"><span class="bold">หว่านเมล็ดพันธุ์ในอัตราที่เหมาะสม</span> คือ 15-20 กิโลกรัม/ไร่ ควรแบ่งแปลงให้มีการระบายถ่ายเทอากาศดี และไม่ควรใส่ปุ๋ยไนโตรเจนสูงเกินไป ถ้าสูงถึง 50 กิโลกรัม/ไร่ โรคไหม้จะพัฒนาอย่างรวดเร็ว</p> \
                                                <p class="indent"><span class="bold">คลุกเมล็ดพันธุ์ด้วยสารป้องกันกำจัดเชื้อรา</span> เช่น ไตรไซคลาโซล (tricyclazone) คาซูกาไมซิน (kasugamycin) คาร์เบนดาซิม (carbendazim) โพรคลอราซ ตามอัตราที่ระบุ</p> \
                                                <p class="indent"><span class="bold">ในแหล่งที่เคยมีโรคระบาดและพบแผลโรคไหม้ทั่วไป 5 เปอร์เซ็นต์ ของพื้นที่ใบ (ในภาพรวม พบเฉลี่ย 2-3 แผลต่อใบ)</span> ควรฉีดพ่นสารป้องกันกำจัดเชื้อรา เช่น ไตรไซคลาโซล (tricyclazone) คาซูกาไมซิน (kasugamycin) อีดิเฟนฟอส ไอโซโพรไทโอเลน (isoprothiolane) คาร์เบนดาซิม (carbendazim) ตามอัตราที่ระบุ</p>'
                                            ,unsafe_allow_html=True)
                                    with tab5:
                                        st.markdown('<p class="indent"> ข้าวพันธุ์สุพรรณบุรี 1 สุพรรณบุรี 60 และชัยนาท 1 ที่ปลูกในภาคเหนือตอนล่าง พบว่าแสดงอาการรุนแรงในบางพื้นที่ และบางปี โดยเฉพาะเมื่อสภาพแวดล้อมเอื้ออำนวย เช่น ฝนพรำ หรือหมอก น้ำค้างจัด อากาศเย็น ใส่ปุ๋ยมากเกินความจำเป็น หรือเป็นดินหลังน้ำท่วม </p>',unsafe_allow_html=True)
                            
                            #TODO Bacteria Blight--------------------------------------------------------------------
                            if 1 in detectedDict:
                                with st.expander(f'ตรวจพบโรคขอบใบแห้ง: {detectedDict[1]} ตำแหน่ง',True):
                                    st.markdown(f'<h4 class="tab-subhead">โรคขอบใบแห้ง (Bacterial Blight Disease)</h4>', unsafe_allow_html=True)
                                    st.markdown('<hr>',unsafe_allow_html=   True)  
                                    tab1,tab2,tab3,tab4= st.tabs(['สาเหตุของโรคและภูมิภาคที่พบ','อาการของโรค','การแพร่ระบาด','การป้องกัน'])
                                    with tab1:  
                                        st.markdown('<h5 class="tab-subhead">สาเหตุของโรค</h5> \
                                                    <p class="indent">เชื้อแบคทีเรีย Xanthomonas oryzae pv. oryzae (ex Ishiyama) Swings et al.</p> \
                                                    <h5 class="tab-subhead">ภูมิภาคที่พบ</h5> \
                                                    <p class="indent">ภาคเหนือ ภาคตะวันออกเฉียงเหนือ และภาคใต้ พบมากในนาน้ำฝน นาชลประทาน</p>'
                                                , unsafe_allow_html=True
                                        )  
                                    with tab2:
                                        st.markdown('<p class="indent">โรคนี้เป็นได้ตั้งแต่ระยะกล้า แตกกอ จนถึง ออกรวง ต้นกล้าก่อนนำไปปักดำจะมีจุดเล็กๆลักษณะช้ำที่ขอบใบของใบล่าง ต่อมาประมาณ 7-10 วัน จุดช้ำนี้จะขยายกลายเป็นทางสีเหลืองยาวตามใบข้าว ใบที่เป็นโรคจะแห้งเร็ว และสีเขียวจะจางลงเป็นสีเทาๆ อาการในระยะปักดำจะแสดงหลังปักดำแล้วหนึ่งเดือนถึงเดือนครึ่ง ใบที่เป็นโรคขอบใบมีรอยขีดช้ำ ต่อมาจะเปลี่ยนเป็นสีเหลือง ที่แผลมีหยดน้ำสีครีมคล้ายยางสนกลมๆขนาดเล็กเท่าหัวเข็มหมุด ต่อมาจะกลายเป็นสีน้ำตาลและหลุดไปตามลม น้ำหรือฝน ซึ่งจะทำให้โรคสามารถระบาดต่อไปได้ แผลจะขยายไปตามความยาวของใบ บางครั้งขยายเข้าไปข้างในตามความกว้างของใบ ขอบแผลมีลักษณะเป็นขอบลายหยัก แผลนี้เมื่อนานไปจะเปลี่ยนเป็นสีเทา ใบที่เป็นโรคขอบใบจะแห้งและม้วนตามความยาว ในกรณีที่ต้นข้าวมีความอ่อนแอต่อโรคและเชื้อโรคมีปริมาณมากเชื้อจะทำให้ท่อน้ำท่ออาหารอุดตัน ต้นข้าวจะเหี่ยวเฉาและแห้งตายทั้งต้นโดยรวดเร็ว เรียกอาการของโรคนี้ว่า ครีเสก (kresek)</p>'
                                            ,unsafe_allow_html=True
                                        )
                                                            
                                    with tab3:
                                        st.markdown('<p class="indent">เชื้อสาเหตุโรคสามารถแพร่ไปกับน้ำ ในสภาพแวดล้อมที่มีความชื้นสูง และสภาพที่มีฝนตก ลมพัดแรง จะช่วยให้โรคแพร่ระบาดอย่างกว้างขวางรวดเร็ว</p>',unsafe_allow_html=True)
                                    with tab4:
                                        st.markdown('<p class="indent"><span class="bold">ใช้พันธุ์ข้าวที่ต้านทาน เช่น พันธุ์สุพรรณบุรี 60 สุพรรณบุรี 90 สุพรรณบุรี 1 สุพรรณบุรี 2 และ กข23</span></p> \
                                                <p class="indent"><span class="bold">ในดินที่อุดมสมบูรณ์ไม่ควรใส่ปุ๋ยไนโตรเจนมาก</span></p> \
                                                <p class="indent"><span class="bold">ไม่ควรระบายน้ำจากแปลงที่เป็นโรคไปสู่แปลงอื่น</span></p> \
                                                <p class="indent"><span class="bold">ควรเฝ้าระวังการเกิดโรคถ้าปลูกข้าวพันธุ์ที่อ่อนแอต่อโรคนี้</span>เช่น พันธุ์ขาวดอกมะลิ 105 กข6 เหนียวสันป่าตอง พิษณุโลก 2  ชัยนาท 1 เมื่อเริ่มพบอาการของโรคบนใบข้าวให้ใช้สารป้องกันกำจัดโรคพืช เช่น ไอโซโพรไทโอเลน คอปเปอร์ไฮดรอกไซด์ เสตร็พโตมัยซินซัลเฟต+ออกซีเตทตราไซคลินไฮโดรคลอร์ไรด์ ไตรเบซิคคอปเปอร์ซัลเฟต</p>'
                                            ,unsafe_allow_html=True)   
                            
                            #TODO Brown Spot--------------------------------------------------------------------
                            if 2 in detectedDict:
                                with st.expander(f'ตรวจพบโรคใบจุดสีน้ำตาล: {detectedDict[2]} ตำแหน่ง',True):
                                    st.markdown(f'<h4 class="tab-subhead">โรคใบจุดสีน้ำตาล (Brown Spot Disease)</h4>', unsafe_allow_html=True)
                                    st.markdown('<hr>',unsafe_allow_html=   True)  
                                    tab1,tab2,tab3,tab4= st.tabs(['สาเหตุของโรคและภูมิภาคที่พบ','อาการของโรค','การแพร่ระบาด','การป้องกัน'])
                                    with tab1:  
                                        st.markdown('<h5 class="tab-subhead">สาเหตุของโรค</h5> \
                                                    <p class="indent">เชื้อรา Bipolaris oryzae Breda de Haan.</p> \
                                                    <h5 class="tab-subhead">ภูมิภาคที่พบ</h5> \
                                                    <p class="indent">ภาคกลาง ภาคเหนือ ภาคตะวันตก ภาคตะวันออกเฉียงเหนือ และภาคใต้ พบมากในนาน้ำฝน นาชลประทาน</p>'
                                                , unsafe_allow_html=True
                                        )  
                                    with tab2:
                                        st.markdown('<p class="indent">แผลที่ใบข้าว พบมากในระยะแตกกอมีลักษณะเป็นจุดสีน้ำตาล รูปกลมหรือรูปไข่ ขอบนอกสุดของแผลมีสีเหลือง มีขนาดเส้นผ่าศูนย์กลาง 0.5-1 มิลลิเมตร แผลที่มีการพัฒนาเต็มที่ขนาดประมาณ 1-2 x 4-10 มิลลิเมตร บางครั้งพบแผลไม่เป็นวงกลมหรือรูปไข่ แต่จะเป็นรอยเปื้อนคล้ายสนิมกระจัดกระจายทั่วไปบนใบข้าว แผลยังสามารถเกิดบนเมล็ดข้าวเปลือก(โรคเมล็ดด่าง) บางแผลมีขนาดเล็ก บางแผลอาจใหญ่คลุมเมล็ดข้าวเปลือก ทำให้เมล็ดข้าวเปลือกสกปรก เสื่อมคุณภาพ เมื่อนำไปสีข้าวสารจะหักง่าย</p>'
                                            ,unsafe_allow_html=True
                                        )
                                                            
                                    with tab3:
                                        st.markdown('<p class="indent">เกิดจากสปอร์ของเชื้อราปลิวไปตามลม และติดไปกับเมล็ด</p>',unsafe_allow_html=True)
                                    with tab4:
                                        st.markdown('<p class="indent bold">ใช้พันธุ์ต้านทานที่เหมาะสมกับสภาพท้องที่ โดยเฉพาะพันธุ์ที่มีคุณสมบัติต้านทานโรคใบสีส้ม</p> \
                                                <p class="indent">- ภาคกลาง ใช้พันธุ์ปทุมธานี 1</p> \
                                                <p class="indent">- ภาคเหนือและภาคตะวันออกเฉียงเหนือ ใช้พันธุ์เหนียวสันป่าตอง และหางยี 71</p> \
                                                <p class="indent"><span class="bold">ปรับปรุงดินโดยการไถกลบฟาง หรือเพิ่มความอุดมสมบูรณ์ดินโดยการปลูกพืชปุ๋ยสด หรือปลูกพืชหมุนเวียนเพื่อช่วยลดความรุนแรงของโรค</span></p> \
                                                <p class="indent"><span class="bold">คลุกเมล็ดพันธุ์ก่อนปลูกด้วยสารป้องกันกำจัดเชื้อรา</span> เช่น แมนโคเซ็บ หรือคาร์เบนดาซิม+แมนโคเซ็บอัตรา 3 กรัม / เมล็ด 1 กิโลกรัม</p> \
                                                <p class="indent"><span class="bold">ใส่ปุ๋ยโปแตสเซียมคลอไรด์ (0-0-60) อัตรา 5-10 กิโลกรัม / ไร่ ช่วยลดความรุนแรงของโรค</span></p> \
                                                <p class="indent"><span class="bold">กำจัดวัชพืชในนา ดูแลแปลงให้สะอาด และใส่ปุ๋ยในอัตราที่เหมาะสม</span></p> \
                                                <p class="indent"><span class="bold">ถ้าพบอาการของโรคใบจุดสีน้ำตาลรุนแรงทั่วไป 10 เปอร์เซ็นต์ของพื้นที่ใบในระยะข้าวแตกกอ หรือในระยะที่ต้นข้าวตั้งท้องใกล้ออกรวง</span> เมื่อพบอาการใบจุดสีน้ำตาลที่ใบธงในสภาพฝนตกต่อเนื่อง อาจทำให้เกิดโรคเมล็ดด่าง ควรพ่นด้วยสารป้องกันกำจัดเชื้อรา เช่น  อีดิเฟนฟอส คาร์เบนดาซิม\
                                                    แมนโคเซ็บ หรือคาร์เบนดาซิม+แมนโคเซบ ตามอัตราที่ระบุ</p>'
                                            ,unsafe_allow_html=True)                                   
                            
                            #TODO Sheath Blight--------------------------------------------------------------------
                            if 3 in detectedDict:
                                with st.expander(f'ตรวจพบโรคกาบใบแห้ง: {detectedDict[3]} ตำแหน่ง',True):
                                    st.markdown(f'<h4 class="tab-subhead">โรคกาบใบแห้ง (Sheath blight Disease)</h4>', unsafe_allow_html=True)
                                    st.markdown('<hr>',unsafe_allow_html=   True)  
                                    tab1,tab2,tab3,tab4= st.tabs(['สาเหตุของโรคและภูมิภาคที่พบ','อาการของโรค','การแพร่ระบาด','การป้องกัน'])
                                    with tab1:  
                                        st.markdown('<h5 class="tab-subhead">สาเหตุของโรค</h5> \
                                                    <p class="indent">เชื้อรา Rhizoctonia solani (Thanatephorus cucumeris (Frank) Donk)</p> \
                                                    <h5 class="tab-subhead">ภูมิภาคที่พบ</h5> \
                                                    <p class="indent">ภาคกลาง ภาคเหนือ และ ภาคใต้ พบมาก ในนาชลประทาน</p>'
                                                , unsafe_allow_html=True
                                        )  
                                    with tab2:
                                        
                                        st.markdown('<p class="indent">เริ่มพบโรคในระยะแตกกอจนถึงระยะใกล้เก็บเกี่ยว ยิ่งต้นข้าวมีการแตกกอมากเท่าใด ต้นข้าวก็จะเบียดเสียดกันมากขึ้น โรคก็จะเป็นรุนแรง ลักษณะแผลสีเขียวปนเทา ขนาดประมาณ 1-4 x 2-10 มิลลิเมตร ปรากฏตามกาบใบ ตรงบริเวณใกล้ระดับน้ำ แผลจะลุกลามขยายใหญ่ขึ้นจนมีขนาดไม่จำกัดและลุกลามขยายขึ้นถึงใบข้าว ถ้าเป็นพันธุ์ข้าวที่อ่อนแอ แผลสามารถลุกลามถึงใบธงและกาบหุ้มรวงข้าว ทำให้ใบและกาบใบเหี่ยวแห้ง ผลผลิตจะลดลงอย่างมากมาย</p>'
                                            ,unsafe_allow_html=True
                                        )
                                                            
                                    with tab3:
                                        st.markdown('<p class="indent">เชื้อราสามารถสร้างเม็ดขยายพันธุ์อยู่ได้นานในตอซังหรือวัชพืชในนาตามดินนา และมีชีวิตข้ามฤดูหมุนเวียนทำลายข้าวได้ตลอดฤดูการทำนา</p>',unsafe_allow_html=True)
                                    with tab4:
                                        st.markdown('<p class="indent"><span class="bold">หลังเก็บเกี่ยวข้าว และเริ่มฤดูใหม่</span> ควรพลิกไถหน้าดินตากแดดเพื่อทำลายเม็ดขยายพันธุ์ของเชื้อราสาเหตุโรค</p> \
                                                <p class="indent"><span class="bold">กำจัดวัชพืชตามคันนาและแหล่งน้ำ</span>เพื่อลดโอกาศการฟักตัวและเป็นแหล่งสะสมของเชื้อราสาเหตุโรค</p>\
                                                <p class="indent"><span class="bold">ใช้ชีวภัณฑ์บาซิลลัส ซับทิลิส (เชื้อแบคทีเรียปฏิปักษ์) ในอัตราที่ระบุ</span></p>\
                                                <p class="indent"><span class="bold">ใช้สารป้องกันกำจัดเชื้อรา</span>เช่น โพรพิโคนาโซล เพนไซคูรอน (25%ดับบลิวพี) หรืออีดิเฟนเฟอส ตามอัตราที่ระบุโดยพ่นสารป้องกันกำจัดเชื้อรานี้ในบริเวณที่เริ่มพบโรคระบาด ไม่จำเป็นต้องพ่นทั้งแปลง เพราะโรคกาบใบแห้งจะเกิดเป็นหย่อมๆ</p>'
                                            ,unsafe_allow_html=True)
                            
                            #TODO Tungro--------------------------------------------------------------------
                            if 4 in detectedDict:
                                with st.expander(f'ตรวจพบโรคใบสีส้ม: {detectedDict[4]} ตำแหน่ง',True):
                                    st.write("โรคโรคใบสีส้ม (Rice Tungro Disease)")
                                    st.markdown(f'<h4 class="tab-subhead">โรคกาบใบแห้ง (Sheath blight Disease)</h4>', unsafe_allow_html=True)
                                    st.markdown('<hr>',unsafe_allow_html=   True)  
                                    tab1,tab2,tab3,tab4= st.tabs(['สาเหตุของโรคและภูมิภาคที่พบ','อาการของโรค','การแพร่ระบาด','การป้องกัน'])
                                    with tab1:  
                                        st.markdown('<h5 class="tab-subhead">สาเหตุของโรค</h5> \
                                                    <p class="indent">เชื้อไวรัส Rice Tungro Bacilliform Virus (RTBV) และ Rice Tungro Spherical Virus (RTSV)</p> \
                                                    <h5 class="tab-subhead">ภูมิภาคที่พบ</h5> \
                                                    <p class="indent">ภาคกลาง และภาคเหนือตอนล่าง พบมากในนาชลประทาน</p>'
                                                , unsafe_allow_html=True
                                        )  
                                    with tab2:
                                        st.markdown('<p class="indent">ต้นข้าวเป็นโรคได้ ทั้งระยะกล้า แตกกอ ตั้งท้อง หากข้าวได้รับเชื้อตอนข้าวอายุอ่อน (ระยะกล้าถึงระยะแตกกอ) ข้าวจะเสียหายมากกว่าได้รับเชื้อตอนอายุแก่ (ระยะตั้งท้องถึงระยะออกรวง) ข้าวเริ่มแสดงอาการหลังจากได้รับเชื้อ 15-20 วัน ทั้งนี้แล้วแต่ว่าข้าวจะได้รับเชื้อระยะใด อาการเริ่มต้นใบข้าวจะเริ่มมีสีเหลืองสลับเขียว ต่อมาจะเปลี่ยนเป็นสีเหลือง เริ่มจากปลายใบเข้าหาโคนใบ ถ้าเป็นรุนแรงในระยะกล้าต้นข้าวอาจถึงตาย ต้นที่เป็นโรคจะเตี้ยแคระแกรน ช่วงลำต้นสั้นกว่าปกติมาก ใบใหม่ที่โผล่ออกมามีตำแหน่งต่ำกว่าข้อต่อใบล่าสุด ถ้าเป็นรุนแรงอาจตายทั้งกอ ถ้าไม่ตายจะออกรวงล่าช้ากว่าปกติ ให้รวงเล็ก หรือไม่ออกรวงเลย</p>'
                                            ,unsafe_allow_html=True
                                        )
                                                            
                                    with tab3:
                                        st.markdown('<p class="indent">แพร่ระบาดได้โดยเพลี้ยจักจั่นสีเขียวซึ่งเป็นแมลงพาหะ</p>',unsafe_allow_html=True)
                                        st.image('tungro_insect.jpeg')
                                    with tab4:
                                        st.markdown('<p class="indent"><span class="bold">ใช้พันธุ์ข้าวต้านทานแมลงเพลี้ยจักจั่นสีเขียว เช่น กข1 กข3</span></p> \
                                                <p class="indent"><span class="bold">กำจัดวัชพืช และพืชอาศัยของเชื้อไวรัสและแมลงพาหะนำโรค</span></p>\
                                                <p class="indent"><span class="bold">ใช้สารป้องกันกำจัดแมลงพาหะ</span> ได้แก่ สารฆ่าแมลงในระยะที่แมลงเป็นตัวอ่อน เช่น ไดโนทีฟูเรน หรือ บูโพรเฟซิน หรือ อีโทเฟนพรอกซ์ ไม่ควรใช้สารฆ่าแมลงผสมกันหลายๆ ชนิด หรือใช้สารฆ่าแมลงผสมสารกำจัดโรคพืชหรือสารกำจัดวัชพืช เพราะอาจทำให้ประสิทธิภาพของสารฆ่าแมลงลดลง ไม่ใช้สารกลุ่มไพรีทรอยด์สังเคราะห์ เช่น ไซเพอร์มิทริน ไซฮาโลทริน เดลต้ามิทริน</p>'
                                            ,unsafe_allow_html=True)
                elif not detectedClass and file_upload != 'กล้องเว็บแคม' : st.markdown('<h4 class="h4-danger">ตรวจไม่พบโรคใบข้าว</h4>', unsafe_allow_html=True)  
    st.session_state.break_cam = False
        
if __name__ == '__main__':
    main()