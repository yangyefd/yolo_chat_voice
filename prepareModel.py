from ultralytics import YOLO
from pathlib import Path

models_dir = Path('./models')
models_dir.mkdir(exist_ok=True)

DET_MODEL_NAME = "yolov8n"

det_model = YOLO(models_dir / f'{DET_MODEL_NAME}.pt')
label_map = det_model.model.names

import json
with open(models_dir / f'{DET_MODEL_NAME}_labels.json', 'w') as f:
    json.dump(label_map, f)

det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=False)

from openvino.preprocess import PrePostProcessor
from openvino.runtime import Type, Layout
from openvino.runtime import serialize
from openvino.runtime import Core

core = Core()
det_ov_model = core.read_model(det_model_path)

#合并预处理
ppp = PrePostProcessor(det_ov_model)
ppp.input(0).tensor().set_shape([1, 640, 640, 3]).set_element_type(Type.u8).set_layout(Layout('NHWC'))
ppp.input(0).preprocess().convert_element_type(Type.f32).convert_layout(Layout('NCHW')).scale([255., 255., 255.])
print(ppp)

model_with_preprocess = ppp.build()
model_with_preprocess_path = str(det_model_path.with_name(f"{DET_MODEL_NAME}_with_preprocess.xml"))
serialize(model_with_preprocess, model_with_preprocess_path)

#语音模型
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('iic/SenseVoiceSmall')

#下完后将模型截切至models文件，默认下载位置为uer/.cache

