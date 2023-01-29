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

#! Page Setup----------------------
st.set_page_config(page_title="Rice Leaf Disease Detection",)
#! Check----------------------
'''
    - ดูภาษาตอน Process  
    - Contrast upload image   
    - blightness image
'''

#!----------------------
modelForWebcam = torch.hub.load('yolov5', 'custom', path='M.pt', _verbose=False, source='local')
#! State------------------------------
if 'disabled_btn' not in st.session_state:
    st.session_state.disabled_btn = False

if 'break_video' not in st.session_state:
    st.session_state.break_video = False

st.cache()
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
                .st-ic > div > div > div > p{font-size:1.8rem;}
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
        diseaseName = ['โรคไหม้','โรคขอบใบแห้ง','โรคใบจุดสีน้ำตาล','โรคกาบใบแห้ง','โรคใบสีส้ม']
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
#! Get Confidence------------------------------------------------------------------------------------------------------------------------
    st.sidebar.header(returnLang('ค่าความเชื่อมั่น','Confidence',lang))
    confidence = st.sidebar.slider(returnLang(
        'เลือกค่าความเชื่อมั่นที่ต้องการ',
        'Select your specific confidence.',
        lang), 
        min_value = 0.0, max_value = 1.0, value = 0.25,
        disabled=st.session_state.disabled_btn    
    )
    modelForWebcam.conf = confidence
#! Get Specific Class------------------------------------------------------------------------------------------------------------------------        
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

#! Get Input Type------------------------------------------------------------------------------------------------------------------------
    st.sidebar.header(returnLang('ประเภทข้อมูลที่ต้องการประมวลผล','Input Type',lang))
    file_upload = st.sidebar.radio(
            label='',
            options=inputOption,
            index=0,
            label_visibility='collapsed',
            key="multi_select",
            disabled=st.session_state.disabled_btn
        )

#! Get Video------------------------------------------------------------------------------------------------------------------------
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
            
#! Get Image------------------------------------------------------------------------------------------------------------------------
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
            pilImg = Image.open(temp_img)
            image_file_name = temp_img.name
            st.write(image_file_name)
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
            enhancer = ImageEnhance.Contrast(pilImg)
            image_file = enhancer.enhance(c_rate)
            st.write(image_file)
            #tffile.write(image_file.read())
            #temp_img = open(tffile.name, 'rb')
        if use_demo_img:
            st.sidebar.warning(
                returnLang('คุณกำลังใช้ภาพตัวอย่างในการประมวลผล',
                    "You're using demo image for processing.",lang
                ), icon="⚠️"
            )
        st.sidebar.image(image_file)

#! Get Webcam------------------------------------------------------------------------------------------------------------------------
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
#! Process Start------------------------------------------------------------------------------------------------------------------------
    start_btn = st.sidebar.button(
        returnLang('เริ่มต้นการประมวลผล','Start Processing',lang), 
        disabled=st.session_state.disabled_btn,
        key='process_btn',
        on_click=disableBtn,
        args=(file_upload, )
    )
    detectedClass = []
    detectedDict = {}
    if start_btn:
        with st.spinner(returnLang(f'กำลังประมวลผล{process_type}...',f'Processing {process_type}...',lang)):
#! Process Image------------------------------------------------------------------------------------------------------------------------
            if file_upload == 'Image' or file_upload == 'รูปภาพ':
                if len(assigned_class_id) > 0:
                    result = modelForWebcam(image_file_name)
                    run(
                        weights='M.pt', 
                        source=image_file_name, 
                        device='cpu', 
                        classes=assigned_class_id,
                        conf_thres=confidence,
                    ) 
                else:
                    result = modelForWebcam(image_file_name)
                    json = result.pandas().xyxy[0].to_dict(orient='list')
                    detectedClass = json['class']
                    detectedDict = {i:detectedClass.count(i) for i in detectedClass}
                if result:
                    r_img = result.render()
                    st.image(r_img)
#! Process Video------------------------------------------------------------------------------------------------------------------------
            if file_upload == 'Video' or file_upload == 'วิดีโอ':
                stop = st.button(
                    'Stop Processing Video...',
                    key='stop_btn',
                    on_click = breakVideo,
                    args = (file_upload, ),
                    type = 'primary'
                )
                run(
                    weights='N.pt', 
                    source=video_path, 
                    device='cpu', 
                    breakVideo=st.session_state.break_video
                ) 
                st.session_state['disabled_btn '] = True
#! Detected Result------------------------------------------------------------------------------------------------------------------------
        with st.container():
            if detectedDict:
                st.empty().success('ตรวจพบโรคใบข้าว')
                with st.empty():
                    with st.container():
                        #TODO Leaf Blast--------------------------------------------------------------------
                        if 0 in detectedDict:
                            with st.expander(f'ตรวจพบโรคไหม้ทั้งหมด: {detectedDict[0]} ตำแหน่ง',True):
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