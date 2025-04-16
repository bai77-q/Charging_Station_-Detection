import argparse
import os
import platform
import sys
from pathlib import Path
import threading
import torch
import moviepy.editor as mp
import multiprocessing
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import gradio as gr
from PIL import Image
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

app = Flask(__name__)


def get_images(vid_name, type, count, frame):
    vid_name = vid_name.split('.')[0]  # 分割字符串并取第一部分
    # 获取脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    save_dir = os.path.join(script_dir, vid_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_filename = f"{save_dir}/{type}_{count}.jpg"  # 定义保存帧的文件名
    cv2.imwrite(frame_filename, frame)  # 保存帧为图像文件

    print(f'Saved {frame_filename}')


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
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
        project=ROOT / 'detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=5,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    count_charge = 0  # 不正常充电柱计数器
    count_fire = 0  # 灭火器计数器
    count_ad = 0  # 小广告计数器
    frame_count = 0  # detect 数量
    save_interval = 50  # 保存间隔
    save_flag = True  # 用来表示是否已经保存了有问题的帧

    for path, im, im0s, vid_cap, s in dataset:

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0][1]
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        for i, det in enumerate(pred):  # per image

            seen += 1
            if webcam:  # batch_size >= 1
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
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

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
            # Stream results处理好之后的带框图片
            im0 = annotator.result()
            vid_name = p.name.split('.')[0]  # 分割字符串并取第一部分
            # 获取脚本所在的绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))

            im_save_dir = os.path.join(script_dir, vid_name)
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            # Save results (image with detections)
            if save_img:
                # 将视频中的有问题的图像保存到本地
                if save_flag:
                    if "不正常" in s:
                        print("检测到不正常充电柱...")
                        get_images(p.name, vid_name, count_charge, im0)
                        count_charge += 1
                        save_flag = False

                    elif "小广告" in s:
                        print("检测到小广告...")
                        get_images(p.name, vid_name, count_ad, im0)
                        count_ad += 1
                        save_flag = False

                    elif "人" in s and "灭火器" in s:
                        print("检测到人使用灭火器...")
                        get_images(p.name, vid_name, count_fire, im0)
                        count_fire += 1
                        save_flag = False
                frame_count += 1
                if not save_flag and frame_count % save_interval == 0:
                    save_flag = True
                if dataset.mode == 'image':

                    # save_path = save_dir / 'frames' / f'{frame_count}_{p.stem}.jpg'
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'

                    # 将处理之后的视频帧存到新视频文件中
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
                        vid_save_path = f"{im_save_dir}/{p.name}"  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(vid_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    print("有问题图片的存储路径为：", im_save_dir)

    return im_save_dir


def parse_opt(images_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp17/weights/best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=images_path, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / '', help='(optional) dataset.yaml path')
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
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    im_save_dir = run(**vars(opt))
    return im_save_dir


# Function to handle file upload and detection

# 指定保存文件的路径

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file):
    # 获取文件的基本名称
    file_name = os.path.basename(file.name)
    # 获取脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'uploaded_files')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = f"{save_dir}/{file_name}"
    if allowed_file(file_name):
        # 检查保存路径是否存在，如果不存在则创建
        # file_name = file_name.lower()
        if file_name.endswith("png") or file_name.endswith(".jpg"):
            print("处理图片")
            image = Image.open(file.name)
            image.save(file_path)  # 保存处理后的图像到文件
        else:
            video_clip = mp.VideoFileClip(file.name)
            video_clip.write_videofile(file_path)  # 保存处理后的视频到文件

        # 执行YOLOv9检测
        opt = parse_opt(save_dir)
        im_save_dir = main(opt)
        name = file_name.split(".")[0]
        zip_filename = f"{script_dir}/{name}.zip"

        # 压缩文件夹为 Zip 文件
        shutil.make_archive(zip_filename[:-4], 'zip', im_save_dir)

        return zip_filename
    else:
        return f"File {file_name} has an invalid extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"

# # Gradio 界面设置
# interface = gr.Interface(
#     fn=save_file,
#     inputs=gr.inputs.File(label="请上传图片"),
#     outputs=[gr.components.File(label="下载处理之后的文件")],
#     title="智慧停车场检测系统-DEMO演示",
#     description="请上传需要检测的图像或者视频，检测结果将在几分钟内完成."
# )
#
# if __name__ == "__main__":
#     interface.launch(server_name='0.0.0.0', debug=True)


import os
import base64
import requests
import gradio as gr
from PIL import Image

# Disable proxies
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# Set environment variables for API access
os.environ["OPENAI_API_KEY"] = "sk-sidDDHsX7zwPtD7K75C129E0177f4d0b9c28F6B1C3Ce1127"
os.environ["OPENAI_API_BASE"] = "https://api.rcouyi.com/v1/chat/completions"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_gpt4_instruction(image_path, task):
    prompt = ""

    if task == "检查充电桩":
        prompt = (
            "请检查以下充电桩图片，并识别是否存在以下问题：\n"
            "1. 识别充电桩充电头状态（充电头摆放在两侧的状态为：正常闲置，如果散落在地面的状态为：不正常摆放，如果正在使用的状态为：正常使用）。\n"
            "2. 充电桩的屏幕是否显示异常信息或故障信息（充电桩中心绿色灯亮为：正常，红色灯亮为：异常，其它颜色灯亮为：其他状态）。\n"
            "3. 充电桩外观是否有明显的污渍或被粘贴小纸条（如果有明显污渍并且有小纸条状态为：存在污渍和纸条，"
            "如果只有污渍状态为：存在污渍，如果只有小纸条状态为：存在小纸条，如果都没有则：正常状态。注意一般充电桩不会存在超过2个二维码，"
            "存在多个二维码可能是有人粘贴广告纸条）。\n"
            "4. 充电桩是否正常工作（如指示灯状态、充电桩电源线是否插入新能源汽车使用等）。\n"
            "5. 其他可能影响充电桩正常使用的情况。\n"
            "请用中文描述每个检查项目的检查结果和建议，要求言简意赅。"
        )
    elif task == "检查灭火器盗窃":
        prompt = (
            "请检查以下图片，并识别是否存在灭火器盗窃的情况：\n"
            "1. 灭火器是否被人为拿出来了。\n"
            "2. 灭火器外观是否有明显损坏。\n"
            "3. 判断是否存在盗窃灭火器的情况（是否是正常检查灭火器还是有人要盗窃灭火器）。\n"
            "请用中文描述每个检查项目的检查结果和建议，要求言简意赅。"
        )
    elif task == "检查防撞梁状态":
        prompt = (
            "请检查以下图片，并识别防撞梁的状态：\n"
            "1. 防撞杆是否存在形变。\n"
            "2. 防撞杆外观是否有明显损坏。\n"
            "3. 防撞杆周围是否有障碍物影响其功能。\n"
            "请用中文描述每个检查项目的检查结果和建议，要求言简意赅。"
        )
    elif task == "通用场景检查":
        prompt = (
            "请检查以下图片，并识别该停车场的状态：\n"
            "1. 车位使用情况：检查车位是否被占用，空闲车位数量。\n"
            "2. 交通标志和标线：检查交通标志、标线是否清晰、完好。\n"
            "3. 照明情况：检查停车场内的照明是否足够，是否有损坏的灯具。\n"
            "4. 安全设施：检查是否有灭火器、安全出口指示等设施，是否完好无损。\n"
            "5. 卫生状况：检查地面是否有垃圾、油渍等，卫生情况是否良好。\n"
            "请用中文描述每个检查项目的检查结果和建议，要求言简意赅。"
        )
    elif task == "其它危险行为检查":
        prompt = (
            "请检查以下图片，并识别停车场的可能危险行为：\n"
            "1. 是否存在违规停车。\n"
            "2. 车辆是否有破损。\n"
            "3. 地面或车辆周围是否有遗留物品。\n"
            "4. 是否有障碍物阻碍通行。\n"
            "5. 是否有人员聚集。\n"
            "6. 是否有设备故障。\n"
            "7. 消防通道是否被占用。\n"
            "8. 是否存在光照不足的情况。\n"
            "9. 是否有可疑活动。\n"
            "请用中文描述每个检查项目的检查结果和建议，要求言简意赅。"
        )

    # OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY")

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(os.environ["OPENAI_API_BASE"], headers=headers, json=payload)
    response_json = response.json()
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response_json)
        return "Error occurred while processing the request."

    chat_response = response_json['choices'][0]['message']['content']

    return chat_response


# Gradio Interface
def process_image(image, task):
    # Save the uploaded image
    image_path = "temp_image.jpg"
    image.save(image_path)

    # Check if image was saved correctly
    if not os.path.exists(image_path):
        return "Error: Image not saved correctly."

    # Get instruction from GPT-4
    instruction = get_gpt4_instruction(image_path, task)

    return instruction


tasks = ["检查充电桩", "检查灭火器盗窃", "检查防撞梁状态", "通用场景检查", "其它危险行为检查"]

with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="请上传图片或视频")
            upload_button = gr.Button("上传并处理")
            upload_output = gr.File(label="下载处理之后的文件")
        with gr.Column():
            image_upload = gr.Image(type="pil", label="请上传图片")
            task_dropdown = gr.Dropdown(choices=tasks, label="选择检查任务")
            gpt4_output_button = gr.Button("获取大模型输出")
            gpt4_output_textbox = gr.Textbox(label="大模型输出结果")

    upload_button.click(save_file, inputs=file_upload, outputs=upload_output)
    gpt4_output_button.click(process_image, inputs=[image_upload, task_dropdown], outputs=gpt4_output_textbox)

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', debug=True)



