import cv2
import numpy as np

# 加载预训练的SSD模型和权重
model_file = "deploy.prototxt"
pretrained_weights = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(model_file, pretrained_weights)

# 读取图像
image = cv2.imread("girls_more.jpg")

# 获取图像尺寸并创建blob
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))

# 输入blob到网络中
net.setInput(blob)
detections = net.forward()

# 遍历检测结果
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]  # 获取置信度

    # 筛选出高置信度的检测结果
    if confidence > 0.5:
        # 计算边界框的坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # 构建标签字符串
        label = f"Face: {confidence * 100:.2f}%"

        # 绘制边界框和标签
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()  # 确保所有窗口都会被关闭
