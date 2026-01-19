import cv2
import numpy as np
from tqdm import trange
import SSD

# 配置路径
input_video = "input_video.avi"  # 将输入路径改为 .avi 文件
output_video = "output_blurred.avi"  # 输出也可以改为 .avi 格式

# 加载 DNN 模型
net = cv2.dnn.readNetFromCaffe('./DNN/deploy.prototxt', './DNN/res10_300x300_ssd_iter_140000.caffemodel')

# 打开视频
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise ValueError("无法打开视频文件，请检查路径是否正确")

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 初始化视频写入器
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 存储模糊帧的列表
blurred_frames = []

# 遍历每一帧
frame_idx = 0
for frame_idx in trange(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理，归一化＋均衡化
    frame_norm = frame / 255.0
    frame_norm = np.uint8(frame_norm * 255)  # 转换回uint8
    yuv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # 获取图像的高和宽
    h, w = frame.shape[:2]

    # 将图像转换为 blob，并通过 DNN 网络进行前向传播
    blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (104.0, 177.0, 123.0), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    # 遍历检测到的区域
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 只处理信心度高的区域（例如大于 0.075）
        if confidence > 0.075:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            face = frame[y:y2, x:x2]
            frame[y:y2, x:x2] = cv2.GaussianBlur(face, (21, 21), 30)  # 使用较小的内核

    # 写入模糊后的视频帧
    out.write(frame)

    # 将模糊帧保存到列表
    blurred_frames.append(frame)

# 释放资源
cap.release()
out.release()

print(f"模糊处理完成！")
print(f"模糊视频保存到: {output_video}")
