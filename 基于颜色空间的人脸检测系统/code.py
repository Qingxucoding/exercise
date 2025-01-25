import copy
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

# 读取图像
filename = 'img7.bmp'
inputpath = './data/' + filename
image = cv.imread(inputpath)
show = cv.cvtColor(copy.deepcopy(image), cv.COLOR_BGR2RGB)
plt.imshow(show)
plt.axis('off')
plt.title("Original Image")
plt.show()


# 转换为 YCbCr 颜色空间
ycbcr_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
# 定义肤色范围
lower_ycbcr = (80, 130, 70)
upper_ycbcr = (255, 180, 120)
# 在生成掩码之前应用高斯模糊
blurred_ycbcr_image = cv.GaussianBlur(ycbcr_image, (5, 5), 0)
# 生成 YCbCr 掩码
mask_ycbcr = cv.inRange(blurred_ycbcr_image, lower_ycbcr, upper_ycbcr)


# 转换为 HSV 空间
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# 定义 HSV 范围
lower_hsv = (0, 20, 60)
upper_hsv = (50, 150, 255)
# 在生成掩码之前应用高斯模糊
blurred_hsv_image = cv.GaussianBlur(hsv_image, (5, 5), 0)
# 生成 HSV 掩码
mask_hsv = cv.inRange(blurred_hsv_image, lower_hsv, upper_hsv)


# 结合 YCbCr 和 HSV 掩码
combined_mask = cv.bitwise_and(mask_ycbcr, mask_hsv)
# 更大或更小的结构元素
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
# 改进后的形态学操作
mask_cleaned = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)  # 填补空洞
mask_cleaned = cv.morphologyEx(mask_cleaned, cv.MORPH_OPEN, kernel)  # 去除噪声


# 定位人脸区域
contours, _ = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
width = max(int(image.shape[0]/400), int(image.shape[1]/400))
for contour in tqdm(contours, desc="Processing contours"):
    if cv.contourArea(contour) > 1000:
        x, y, w, h = cv.boundingRect(contour)
        # 添加宽高比约束
        aspect_ratio = w / h
        if 0.75 < aspect_ratio < 1.5:  # 假定人脸区域接近正方形
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), width)


# 显示及保存结果
image_rgb = image
outputpath = './result/YCbCr+HSV/' + filename
cv.imwrite(outputpath, image_rgb)
image_rgb = cv.cvtColor(image_rgb, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Detected Image (YCbCr + HSV)")
plt.show()
