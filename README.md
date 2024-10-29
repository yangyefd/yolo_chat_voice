# 鸣谢
本仓库基于https://github.com/AJV009/explore-scene-w-object-detection，
感谢作者的开源，暂未找到许可证，若违反许可请及时与我联系。

# 功能特性
1. 未使用本地LLM大模型（无法function call，且回答错误过多）
2. zhipu api 支持function call模式调用检测
3. 支持语音进行物体检测，不过目前语音模块未进行多线程处理，开启后会长期占用处理器
4. 讲故事会较多消耗LLM，导致反应时间过长，且程序会长时间处于无响应状态，应当给予用户反馈界面优化使用体验
5. 语音模块开启后支持TTS，即支持语音播报功能，方便各类用户使用

# 使用说明：
先使用prepareModel下载模型：
1. 需要下载yolo模型，py会将其变为openvino格式且不进行PTQ量化，故不需要使用数据集进行校正。
2. 需下载sensevoice模型进行语音识别，将语音转为文字进而由LLM进行处理
3. 需准备glm api key，此程序使用的glm-4-flash模型为免费模型，故请大胆提供api key

