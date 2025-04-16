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

    # OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY")

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print(prompt)

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
        "max_tokens": 300
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


# Define tasks
tasks = ["检查充电桩", "检查灭火器盗窃", "检查防撞梁状态", "通用场景检查"]

# Create Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil"), gr.Dropdown(choices=tasks, label="选择检查任务")],
    outputs="text",
    title="智慧停车场安全检查",
    description="上传图片并选择检查任务，获取详细检查结果和建议。"
)

# Launch the interface with global access
interface.launch()
