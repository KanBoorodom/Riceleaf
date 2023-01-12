# YOLOv5 🚀 by Ultralytics, GPL-3.0 license\
# This is Goodone ==================
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
from curses import has_key
import streamlit as st
import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False, #!------ diff  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  #!------ diff # video frame-rate stride
        breakVideo=False
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    #! Streamlit Var---------------------
    detectedClass = dict()
    showVideo = st.empty()
    success = st.empty()
    expanderContainer = st.empty()
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size #!------ diff
    if webcam:
        view_img = check_imshow(warn=True) #!------ diff
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) #break_cam=breakVideo
        if breakVideo: return
        bs = len(dataset)
    elif screenshot: #!------ diff
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile()) #!------ diff
    for path, im, im0s, vid_cap, s in dataset:
        if(breakVideo): return
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference #!------ diff
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS #!------ diff
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                #return
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                names_ = []
                cnt = []
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    names_.append(names[int(c)])
                    cnt.append(int(n.detach().cpu().numpy()))
                #! Collect Class Detect--------------------------------
                detectedClass.update(dict(zip(names_, cnt)))
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            #! Streamlit -----------------------------------------------------
            with st.container():
                showVideo.image(im0, channels="BGR", use_column_width=True)
                if detectedClass:
                    with expanderContainer:
                        with st.container():
                            st.markdown('<h4 class="h4-success">ตรวจพบโรคใบข้าว</h4>', unsafe_allow_html=True)  
                            #! Leaf Blast----------------------------------
                            if 'Leaf Blast' in detectedClass:
                                with st.expander(f'ตรวจพบโรคใบไหม้ทั้งหมด: {detectedClass["Leaf Blast"]} ตำแหน่ง',True):
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
                            #!ฺBacteria Blight----------------------------------
                            if 'Bacteria Blight' in detectedClass:
                                with st.expander(f'ตรวจพบโรคขอบใบแห้ง: {detectedClass["Bacteria Blight"]} ตำแหน่ง',True):
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
                            #!Brown Spot----------------------------------
                            if 'Brown Spot' in detectedClass:
                                with st.expander(f'ตรวจพบโรคใบจุดสีน้ำตาล: {detectedClass["Brown Spot"]} ตำแหน่ง',True):
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
                            
                            #!Sheath Blight----------------------------------
                            if 'Sheath Blight' in detectedClass:
                                with st.expander(f'ตรวจพบโรคกาบใบแห้ง: {detectedClass["Sheath Blight"]} ตำแหน่ง',True):
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
                            
                            #!Tungro----------------------------------
                            if 'Tungro' in detectedClass:
                                with st.expander(f'ตรวจพบโรคใบสีส้ม: {detectedClass["Tungro"]} ตำแหน่ง',True):
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
                    st.header('ไม่พบโรคใบข้าวจากการประมวลผล')
                    st.error('การประมวลผลไม่พบอาจเกิดจากการเลือกใช้ค่าความเชื่อมั่นที่สูงมากจนเกินไป หรือการใช้รูปภาพที่ไม่เหมาะสม')
             #if(breakVideo): break
            if view_img: #!------ diff
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        #exit()
        # Print time (inference-only)
        #if(breakVideo): break
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    # Print results
    if breakVideo:
        exit()
    if not breakVideo: 
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
