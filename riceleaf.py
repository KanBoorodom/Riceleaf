import streamlit as st
from streamlit_webrtc import webrtc_streamer
import streamlit.components.v1 as components
import av
import tempfile
import numpy as np
from PIL import Image, ImageEnhance
import torch
from yolov5.detect import run
from showresult import showResult
import queue
st.cache()
#! Page Setup----------------------
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    initial_sidebar_state= 'auto' 
)
#! RTC Configuration----------------------
rtc_configuration = {
    'iceServers': [
        {'urls':['stun:stun.l.google.com:19302']}
    ]
}
modelForImage = torch.hub.load('yolov5', 'custom', path='weight/best (15).pt', _verbose=False, source='local')
modelForWebcam = torch.hub.load('yolov5', 'custom', path='weight/N.pt', _verbose=False, source='local')

detectedClass = []
detectedDict = {}
#! State------------------------------
if 'use_program' not in st.session_state:
    st.session_state.use_program = False
if 'view_ricedata' not in st.session_state:
    st.session_state.view_ricedata = False
result_queue = (queue.Queue())
def predictWebcam(frame):
    global detectedDict
    img = frame.to_rgb()
    img = img.to_ndarray() #Converet video frame to array
    results = modelForWebcam(img, size=(300,300))
    bbox_img = np.array(results.render()[0])
    result_queue.put(results)
    return av.VideoFrame.from_ndarray(bbox_img)
    """ 
    json = results.pandas().xyxy[0].to_dict(orient='list')
    detectedClass = json['class']
    detectedDict = {i:detectedClass.count(i) for i in detectedClass} 
    bbox_img = np.array(results.render()[0])
    return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")"""
def chooseUseProgram():
    st.session_state.use_program = True
    st.session_state.view_ricedata = False
def chooseViewRicedata():
    st.session_state.use_program = False
    st.session_state.view_ricedata = True
def main():
#!---- Styling-------------
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    diseaseName = ['โรคใบไหม้','โรคขอบใบแห้ง','โรคใบจุดสีน้ำตาล','โรคกาบใบแห้ง','โรคใบสีส้ม']
    inputOption = ('รูปภาพ', 'วิดีโอ', 'กล้องถ่ายรูป','กล้องเว็บแคม')
    title = st.empty()
    title.title('โปรแกรมตรวจสอบโรคใบข้าวด้วย YOLOV5')
    if not st.session_state.use_program and not st.session_state.view_ricedata: #TODO default when start program -----
        st.button('ใช้โปรแกรมตรวจสอบโรคใบข้าว', on_click=chooseUseProgram)
        st.button('ดูข้อมูลโรคข้าว', on_click=chooseViewRicedata)
    elif not st.session_state.use_program:
        st.button('ใช้โปรแกรมประมวลผลโรคใบข้าว', on_click=chooseUseProgram)
        title.title('ข้อมูลโรคข้าว(แบ่งตามหมวดหมู่สาเหตุการเกิดโรค)')
    if st.session_state.use_program:
        st.markdown('''
        <a class="toggle" href="javascript:document.getElementsByClassName('css-4l4x4v edgvbvh3')[1].click();" target="_self">ตั้งค่าโปรแกรม</a>
        ''', unsafe_allow_html=True)
        if not st.session_state.view_ricedata:
            st.sidebar.button('ดูข้อมูลโรคข้าว', on_click=chooseViewRicedata)
        st.sidebar.markdown(f'<h4 class="option">ตั้งค่าโปรแกรม</h4>', unsafe_allow_html=True) 
        #! Get Confidence------------------------------------------------------------------------------------------------------------------------
        st.sidebar.header('ค่าความเชื่อมั่น')
        confidence = st.sidebar.slider(
            'เลือกค่าความเชื่อมั่นที่ต้องการ',
            min_value = 0.0, max_value = 1.0, value = 0.354,
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
            )

        #! Get Video------------------------------------------------------------------------------------------------------------------------
        if file_upload == 'วิดีโอ':
            process_type = 'วิดีโอ'
            video_file_buffer = st.sidebar.file_uploader(
                'เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบด้านล่าง', 
                type=['mp4', 'mov', 'avi', 'asf', 'm4v'],
            )
            tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            if not video_file_buffer:
                demo_video = 'testBS.mov'
                st_video = open(demo_video, 'rb')
                video_path = st_video.name
                st.sidebar.warning('คุณกำลังใช้วิดีโอตัวอย่างในการประมวลผล', icon="⚠️")
            else:
                tffile.write(video_file_buffer.read())
                st_video = open(tffile.name, 'rb')
                video_path = st_video.name
            emptyVideo = st.empty()
            st.sidebar.video(st_video)
            emptyVideo.video(st_video)
                
        #! Get Image------------------------------------------------------------------------------------------------------------------------
        elif file_upload == 'รูปภาพ':
            process_type = 'รูปภาพ'
            image_file = st.sidebar.file_uploader(
                'เลือก browse files หรือลากไฟล์ที่ต้องการไปที่กรอบด้านล่าง', 
                type=['jpeg','jpg','png','webp'],
            )
            tffile = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            if not image_file: 
                image_file = Image.open('3CLASS.png')
                pilImg = image_file
                image_file = image_file.filename
                st.sidebar.warning('คุณกำลังใช้ภาพตัวอย่างในการประมวลผล', icon="⚠️")
            else: 
                tffile.write(image_file.read())
                temp_img = open(tffile.name, 'rb')
                pilImg = Image.open(temp_img)
                image_file = temp_img.name
            st.sidebar.header('รูปแบบของการปรับปรุงรูปภาพ','Enhancing Image')
            enhance_type = st.sidebar.radio('', 
                ['Original (ภาพดั้งเดิม)','Contrast (ปรับความคมชัดของสีในภาพ)','Brightness (ปรับความสว่างของภาพ)'],
                label_visibility='collapsed'
            )
            if enhance_type == 'Contrast (ปรับความคมชัดของสีในภาพ)':
                c_rate = st.sidebar.slider('Contrast',0.5,3.5, value=1.0)
                st.sidebar.error('** การปรับค่ารูปแบบของรูปภาพอาจส่งผลต่อความแม่นยำในการประมวลผลได้')
                enhancer = ImageEnhance.Contrast(pilImg)
                image_file = enhancer.enhance(c_rate)
            if enhance_type == 'Brightness (ปรับความสว่างของภาพ)':
                b_rate = st.sidebar.slider('Brightness',0.5,3.5, value=1.0)
                st.sidebar.error('** การปรับค่าความสว่างของรูปภาพอาจส่งผลต่อความแม่นยำในการประมวลผลได้')
                enhancer = ImageEnhance.Brightness(pilImg)
                image_file = enhancer.enhance(b_rate)
            st.sidebar.image(image_file)
            showImage = st.empty()
            showImage.image(image_file)

        #! Get Webcam Image Capture------------------------------------------------------------------------------------------------------------------------
        elif file_upload == 'กล้องถ่ายรูป':
            img_capture = st.camera_input("", label_visibility='collapsed') 
            if not img_capture:
                components.html('''
                    <script>
                        const strlit = window.parent.document;
                        let btnClass = strlit.querySelector('.ejtjsn20')
                        btnClass.innerHTML = 'ถ่ายรูปเพื่อประมวลผล';
                    </script>
                    ''', height=5,)
            if img_capture: 
                components.html('''
                <script>
                    const strlit = window.parent.document;
                    let btnClass = strlit.querySelector('.ejtjsn20')
                    btnClass.innerHTML = 'ถ่ายรูปใหม่';
                </script>
                ''', height=5,)
                st.markdown('----')
                st.markdown(f'<h4>ผลการประมวลผลภาพถ่าย:</h4>', unsafe_allow_html=True) 
                camera = st.empty()
                process_type = 'กล้องถ่ายรูป'
                tffile = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                tffile.write(img_capture.read())
                temp_img = open(tffile.name, 'rb')
                pilImg = Image.open(temp_img)
                image_file = temp_img.name
                camera.image(img_capture)  
        #! Get Webcam------------------------------------------------------------------------------------------------------------------------
        elif file_upload == 'กล้องเว็บแคม':
            process_type = 'กล้องเว็บแคม'
        st.sidebar.markdown('---')

        #! Process Start------------------------------------------------------------------------------------------------------------------------
        #TODO Sidebar----
        if file_upload != 'กล้องเว็บแคม' and file_upload != 'กล้องถ่ายรูป':
            start_btn = st.sidebar.button('เริ่มต้นการประมวลผล', key='process_btn')
        if file_upload == 'รูปภาพ' or file_upload == 'วิดีโอ':
            st.write('-----')
            emptyStartBtn = st.empty()
            start_mobile_btn = emptyStartBtn.button(
                'เริ่มต้นการประมวลผล',
                key='process_btn1'
            )
        st.sidebar.markdown('''
        <a class="toggle" href="javascript:document.getElementsByClassName('css-4l4x4v edgvbvh3')[1].click();" target="_self">สิ้นสุดการตั้งค่าโปรแกรม</a>
        ''', unsafe_allow_html=True)
        st.write(start_btn)
        st.write(start_mobile_btn)
        if  file_upload == 'กล้องเว็บแคม' or \
            (file_upload == 'กล้องถ่ายรูป' and img_capture) or \
            (file_upload != 'กล้องถ่ายรูป' and start_btn) or \
            (file_upload != 'กล้องถ่ายรูป' and start_mobile_btn):
            with st.spinner(f'กำลังประมวลผล{process_type}...'):

        #! Process Image------------------------------------------------------------------------------------------------------------------------
                if file_upload == 'รูปภาพ' or file_upload == 'กล้องถ่ายรูป':
                    if len(assigned_class_id) > 0:
                        modelForImage.classes = assigned_class_id
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
                elif file_upload == 'วิดีโอ':
                    stop = st.button(
                        'Stop Processing Video...',
                        key='stop_btn',
                        type = 'primary'
                    )
                    emptyVideo.empty()
                    emptyStartBtn.empty()
                    if not stop:
                        if len(assigned_class_id) > 0:
                            run(
                                weights='weight/N-Last.pt', 
                                source=video_path, 
                                device='cpu', 
                                conf_thres=confidence,
                                classes=assigned_class_id,
                            ) 
                        else:
                            run(
                                weights='weight/N-Last.pt', 
                                source=video_path, 
                                device='cpu', 
                                conf_thres=confidence,
                            )  

        #! Process Webcam Camera------------------------------------------------------------------------------------------------------------------------
                elif file_upload == 'กล้องเว็บแคม':          
                    showStreamResult = st.sidebar.checkbox('แสดงข้อมูลโรคใบข้าวระหว่างประมวลผลด้วยกล้องเว็บแคม')
                    if showStreamResult: st.sidebar.warning('โปรดระวัง!! ข้อมูลที่แสดงผลระหว่างการประมวลผลด้วยกล้องเว็บแคมอาจมีความไม่ต่อเนื่อง', icon="⚠️")
                    translations={
                        "start": "เริ่มต้นการประมวลผล",
                        "stop": "หยุดการประมวลผล",
                        "select_device": "เลือกอุปกรณ์",
                    }
                    stream = webrtc_streamer(
                        key="webcam_process", 
                        video_frame_callback=predictWebcam,
                        rtc_configuration=rtc_configuration,
                        media_stream_constraints={"video": True, "audio": False},
                        translations=translations,
                        async_processing=True
                    )
                    if showStreamResult and stream.state.playing:
                        emptyCamResult = st.empty()
                        while True:
                            try:
                                result = result_queue.get(timeout=1.0)        
                            except queue.Empty:
                                result = {}                     
                            if result != {}:
                                json = result.pandas().xyxy[0].to_dict(orient='list')
                                detectedClass = json['class']
                                detectedDict = {i:detectedClass.count(i) for i in detectedClass}  
                                with emptyCamResult:
                                    showResult(detectedDict, detectedClass, file_upload)
                            else:
                                emptyCamResult.write('no detected')
                                                
        #! Detected Result------------------------------------------------------------------------------------------------------------------------
            if (file_upload == 'รูปภาพ' and (start_btn or start_mobile_btn)) or (file_upload == 'กล้องถ่ายรูป' and img_capture):
                with st.container():
                    showResult(detectedDict,detectedClass, file_upload)
    elif st.session_state.view_ricedata: showResult(showDetected=False)
if __name__ == '__main__':
    main()