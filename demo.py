from openvino.runtime import Core
core = Core()
# Load the pre optimized model
yolov8n_with_preprocess_model = core.read_model('./models/yolov8n_openvino_model/yolov8n_with_preprocess.xml',)
API_KEY = None

import json
# Load the label map
with open('models/yolov8n_labels.json', 'r') as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

# Camera source could be different for different systems, please test with `run_object_detection` function
CAMERA_SOURCE = 0
# distance_confidence_matrix, all_detected_objects = run_object_detection(source=0, flip=True, model=yolov8n_with_preprocess_model, label_map=label_map, core=core, device="AUTO", interval=500)

def view_camera_remover(all_detected_objects):
    if 'view_camera' in all_detected_objects:
        # remove the view camera from the list of detected objects
        all_detected_objects.remove('view_camera')
    return all_detected_objects

def identify_object(object_name, distance_confidence_matrix, all_detected_objects):
    all_detected_objects = view_camera_remover(all_detected_objects)
    if len(all_detected_objects) != 0:
        return "告诉用户检测到了什么: " + ", ".join(all_detected_objects)
    return "未发现物体"

import os
from zhipuai import ZhipuAI
import json


from utils.objectDetect import run_object_detection


label_map_values = list(label_map.values())

# model = "gpt-3.5-turbo-0613"

functions = [
        {
            "type": "function",
            "function":{
                "name": "identify_object",
                "description": "Identify all the objects present in the scene",
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            }
        }
    ]

# Define available functions names
available_functions = {
    "identify_object": identify_object
}

# Chat history
chat_history = [
    {
        "role": "system", 
        "content": """ 你需要帮助小孩子或者帮助视障人士辨认物体，你可以使用yolo进行物体辨认"""
    }
]

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel, QDialog, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QMovie, QPixmap
from PyQt5.QtWidgets import QSizePolicy
import markdown2
import pyaudio
import wave
import webrtcvad
import numpy as np
import pyaudio
import webrtcvad
import numpy as np
import wave
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import pyttsx3
import os

model_dir = "./models/iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    disable_update=True
)

class LoginWindow(QDialog):  # 改为继承自QDialog
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题
        self.setWindowTitle('APIKEY')
        
        # 布局管理器
        layout = QVBoxLayout()
        
        # 密码标签与输入框
        self.password_label = QLabel('密码:', self)
        self.password_edit = QLineEdit(self)
        self.password_edit.setEchoMode(QLineEdit.Password)  # 设置密码模式
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_edit)

        # 登录按钮
        self.login_button = QPushButton('登录', self)
        self.login_button.clicked.connect(self.handle_login)
        layout.addWidget(self.login_button)

        # 将布局添加到窗口
        self.setLayout(layout)
        self.resize(1600, 1200)

    def handle_login(self):
        password = self.password_edit.text()
        global API_KEY
        API_KEY = password

        self.accept()  # 成功登录


class ChatWidget(QWidget):
    def __init__(self):
        super().__init__()

        global API_KEY
        self.LLM_model = "glm-4-flash"
        self.client = ZhipuAI(api_key=API_KEY) # 请填写您自己的APIKey

        # 初始化聊天历史记录
        self.chat_history = []

        # 创建输入框
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("Type your message here, press enter to send.")
        self.input_text.returnPressed.connect(self.send_message)

        # 创建输出文本框
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 创建 QLabel 用于显示图像
        self.image_label = QLabel(self)
        # self.image_label.setAlignment(Qt.AlignCenter)  # 图像居中对齐
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 声音开关
        self.toggle_button = QPushButton('Voice Off', self)
        self.toggle_button.setCheckable(True)  # 使按钮可被选中（保持按下状态）
        self.toggle_button.clicked.connect(self.on_toggle_button_clicked)


        # 创建清除按钮
        self.clear_button = QPushButton("Clear the chat history")
        self.clear_button.clicked.connect(self.clear_chat)

        # 创建解释按钮
        self.explain_button = QPushButton("Explain all objects")
        self.explain_button.clicked.connect(self.explain_objects)

        # 创建故事按钮
        self.story_button = QPushButton("Tell me a fun story about the objects")
        self.story_button.clicked.connect(self.tell_story)

        # 设置布局
        layout = QVBoxLayout()

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_text)
        input_layout.addWidget(self.clear_button)

        image_voice_layout = QVBoxLayout()
        image_voice_layout.addWidget(self.image_label)
        image_voice_layout.addWidget(self.toggle_button)

        # 创建水平布局
        horizontal_layout = QHBoxLayout()
        # 将文本框和图像标签添加到水平布局
        horizontal_layout.addLayout(image_voice_layout)
        horizontal_layout.addWidget(self.output_text)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.explain_button)
        button_layout.addWidget(self.story_button)
        
        
        # 将水平布局添加到主布局
        layout.addLayout(horizontal_layout)
        # layout.addLayout(image_layout)
        # layout.addWidget(self.output_text)
        layout.addLayout(input_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(1600, 1200)

        self.display_image('images/frame_init.jpg')

    def on_toggle_button_clicked(self, checked):
        if checked:
            self.toggle_button.setText('Voice On')
            self.listen(listen_enable=True)
            print("Button is On")
        else:
            self.toggle_button.setText('Voice off')
            self.listen(listen_enable=False)
            print("Button is Off")

    def listen(self,listen_enable=True):
        # 音频参数
        FORMAT = pyaudio.paInt16  # 采样位数（16位）
        CHANNELS = 1  # 单声道
        RATE = 16000  # 采样率（16kHz，VAD通常使用这个采样率）
        CHUNK = 320  # 每个缓冲区的帧数（16000 Hz / 50 Hz = 320 samples）
        RECORD_SECONDS = 1
        WAVE_OUTPUT_FILENAME = "output.wav"  # 输出文件名
        one_second_num = RATE / CHUNK

        # 初始化VAD
        vad = webrtcvad.Vad()
        vad.set_mode(1)  # 设置VAD的敏感度，0-3，数字越大越敏感

        # 初始化PyAudio
        p = pyaudio.PyAudio()

        # 打开流
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("开始监听... ")
        frames = []
        while listen_enable:
            no_speak = 0
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                # 读取音频数据
                data = stream.read(CHUNK)
                
                # 将字节数据转换为numpy数组
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # 使用VAD检测语音活动
                is_speech = vad.is_speech(data, RATE)
                
                if is_speech:
                    print("speak")
                    frames.append(data)
                else:
                    print("no speak")
                    print(len(frames))
                    no_speak += 1
                    if no_speak > 30:
                        frames = []
                        break
            if len(frames) > (RATE / CHUNK * RECORD_SECONDS):
                # 停止并关闭流
                stream.stop_stream()
                stream.close()
                p.terminate()

                # 保存录制的音频到WAV文件
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                # en
                res = model.generate(
                    input="output.wav",
                    cache={},
                    language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,  #
                    merge_length_s=15,
                )
                text = rich_transcription_postprocess(res[0]["text"])
                print(text)
                
                frames = []
                p = pyaudio.PyAudio()
                # 打开流
                stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)
                
                if "你好" in text or "助手" in text:
                    #输出你好
                    pp = pyttsx3.init()
                    pp.say('您好，请说')
                    pp.runAndWait()

                    no_speak_t = 0
                    for i in range(0, int(RATE / CHUNK * 60)):
                        # 读取音频数据
                        data = stream.read(CHUNK)
                        
                        # 将字节数据转换为numpy数组
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        # 使用VAD检测语音活动
                        is_speech = vad.is_speech(data, RATE)
                        
                        if is_speech:
                            print("正在说话")
                            frames.append(data)
                        else:
                            no_speak_t += 1
                            print("没有说话")
                            if no_speak_t > RATE / CHUNK * 5:
                                break
                    if len(frames) > (RATE / CHUNK * 2):
                        # 停止并关闭流
                        stream.stop_stream()
                        stream.close()
                        p.terminate()

                        # 保存录制的音频到WAV文件
                        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                        wf.close()

                        # en
                        res = model.generate(
                            input="output.wav",
                            cache={},
                            language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                            use_itn=True,
                            batch_size_s=60,
                            merge_vad=True,  #
                            merge_length_s=15,
                        )
                        question = rich_transcription_postprocess(res[0]["text"])
                        print(question)

                        #调用大模型
                        # 显示用户的问题
                        self.display_message(f"User: {question}")

                        messages = [
                            {
                                "role": "user",
                                "content": f"{question}"
                            }
                        ]
                        
                        function_chat_history = chat_history.copy()
                        function_chat_history.append({"role": "user", "content": str(question)})
                        response_message = self.chat_wrapper(function_chat_history, functions=functions)
                        if response_message.tool_calls:
                            function_name = response_message.tool_calls[0].function.name
                            function_to_call = available_functions[function_name]
                            function_args = json.loads(response_message.tool_calls[0].function.arguments)
                            distance_confidence_matrix, all_detected_objects = run_object_detection(source=CAMERA_SOURCE, flip=True, model=yolov8n_with_preprocess_model, label_map=label_map, core=core, device="AUTO", interval=500)
                            function_response = function_to_call(
                                object_name=function_args.get("object_name"),
                                distance_confidence_matrix=distance_confidence_matrix,
                                all_detected_objects=all_detected_objects
                            )
                            self.display_image()
                            chat_history.append({"role": "user", "content": str(question)})

                            chat_history.append({"role": "user", "content": str(function_response)})

                        # 在这里添加与API交互的逻辑
                        response = self.chat_wrapper(chat_history).content

                        # 显示助手的回答
                        self.display_message(f"Assistant: {response}")
                        pp = pyttsx3.init()
                        pp.say(response)
                        pp.runAndWait()


                        frames = []
                        p = pyaudio.PyAudio()
                        # 打开流
                        stream = p.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK)

        print("停止监听... ")
        # 停止并关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()

    def send_message(self):
        # 获取用户输入
        question = self.input_text.text().strip()
        if not question:
            return

        # 清空输入框
        self.input_text.clear()

        # 显示用户的问题
        self.display_message(f"User: {question}")

        messages = [
            {
                "role": "user",
                "content": f"{question}"
            }
        ]
        
        function_chat_history = chat_history.copy()
        function_chat_history.append({"role": "user", "content": str(question)})
        response_message = self.chat_wrapper(function_chat_history, functions=functions)
        if response_message.tool_calls:
            function_name = response_message.tool_calls[0].function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message.tool_calls[0].function.arguments)
            distance_confidence_matrix, all_detected_objects = run_object_detection(source=CAMERA_SOURCE, flip=True, model=yolov8n_with_preprocess_model, label_map=label_map, core=core, device="AUTO", interval=500)
            function_response = function_to_call(
                object_name=function_args.get("object_name"),
                distance_confidence_matrix=distance_confidence_matrix,
                all_detected_objects=all_detected_objects
            )
            # chat_history.append({"role": "function", "name": function_name, "content": function_response})
            chat_history.append({"role": "user", "content": str(question)})
            chat_history.append({"role": "user", "content": str(function_response)})

        # 在这里添加与API交互的逻辑
        response = self.chat_wrapper(chat_history).content

        # 显示助手的回答
        self.display_message(f"Assistant: {response}")

    def clear_chat(self):
        # 清除聊天历史
        self.output_text.clear()
        self.chat_history.clear()

    def explain_objects(self):
        # 在这里添加物体检测逻辑
        distance_confidence_matrix, all_detected_objects = run_object_detection(source=CAMERA_SOURCE, flip=True, model=yolov8n_with_preprocess_model, label_map=label_map, core=core, device="AUTO", interval=500)
        all_detected_objects = view_camera_remover(all_detected_objects)
        all_objects = ", ".join(all_detected_objects)
        query = f"告诉我这些物体的详情: {all_objects}"
        self.display_image()
        self.button_eventhandler(query)
        pass

    def tell_story(self):
        # 在这里添加物体检测逻辑
        distance_confidence_matrix, all_detected_objects = run_object_detection(source=CAMERA_SOURCE, flip=True, model=yolov8n_with_preprocess_model, label_map=label_map, core=core, device="AUTO", interval=500)
        all_detected_objects = view_camera_remover(all_detected_objects)
        all_objects = ", ".join(all_detected_objects)
        query = f"讲个关于这些物体的故事: {', '.join(all_detected_objects)}"
        self.display_image()
        self.button_eventhandler(query)
        pass

    def display_message(self, message):
        # 将消息追加到输出框
        html_text = markdown2.markdown(message)
        self.output_text.append(html_text)
        # self.output_text.setHtml(message)
    def display_image(self,file_path=None):
        frame_init_flag = 0
        if file_path is not None:
            frame_init_flag = 1
            image_path = file_path
        else:
            image_path = 'images/frame_last.jpg'
        # 加载图片
        self.pixmap = QPixmap(image_path)
        
        if not self.pixmap.isNull():
            # 如果图片太大，进行缩放
            self.update_image_size()
            if frame_init_flag == 0:
                os.remove(image_path)
        else:
            print(f"无法加载图片: {image_path}")

    def update_image_size(self):
        # 根据当前 QLabel 的大小调整图像大小
        scaled_pixmap = self.pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        # 当窗口大小变化时，更新图像大小
        self.update_image_size()
        super().resizeEvent(event)

    def chat_wrapper(self, messages, functions=[]):
        if functions != []:
            response = self.client.chat.completions.create(
                model=self.LLM_model, # 请填写您要调用的模型名称
                messages=messages,
                tools=functions,
                tool_choice="auto",
            )
            return response.choices[0].message
        else:
            response = self.client.chat.completions.create(
                model=self.LLM_model, # 请填写您要调用的模型名称
                messages=messages,
            )
            return response.choices[0].message

    def button_eventhandler(self,message):
        chat_history.append({"role": "user", "content": str(message)})
        answer = self.chat_wrapper(chat_history).content
        answer_formatted = answer
        chat_history.append({"role": "assistant", "content": str(answer_formatted)})
        # 显示助手的回答
        self.display_message(f"Assistant: {answer_formatted}")

def main():
    app = QApplication(sys.argv)
    
    login = LoginWindow()
    if login.exec_() == QDialog.Accepted:  # 使用 exec_() 显示对话框并等待关闭
        chat_widget = ChatWidget()
        chat_widget.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)

# 主函数
if __name__ == '__main__':
    main()
