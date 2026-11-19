

import cv2
import os
import numpy as np


RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_photos")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed_test1")
OUTPUT_SIZE = (64, 64)  
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"



def detect_and_crop_face(image_path, output_size):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [警告] 无法读取图片: {image_path}")
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print(f"  [提示] 未检测到人脸，直接使用整图: {os.path.basename(image_path)}")
        resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
        return resized

    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        print(f"  [提示] 检测到 {len(faces)} 张人脸，取最大的一张")

    x, y, w, h = faces[0]

    margin = int(0.1 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    cropped = gray[y1:y2, x1:x2]

    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)

    return resized


def main():
    print("=" * 60)
    print("人脸预处理程序启动")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    total_success = 0
    total_fail = 0

    person_folders = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])

    if not person_folders:
        print(f"\n[错误] 在 {RAW_DIR} 中没有找到任何人物文件夹！")
        print("请先在 raw_photos/ 下创建 Person_01 ~ Person_15 文件夹，并放入照片。")
        return

    print(f"\n找到 {len(person_folders)} 个人物文件夹")

    for person_name in person_folders:
        raw_person_dir = os.path.join(RAW_DIR, person_name)
        processed_person_dir = os.path.join(PROCESSED_DIR, person_name)
        os.makedirs(processed_person_dir, exist_ok=True)

        print(f"\n--- 处理 {person_name} ---")

        # 支持的图片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        image_files = sorted([
            f for f in os.listdir(raw_person_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if not image_files:
            print(f"  [警告] {person_name} 文件夹中没有图片")
            continue

        success_count = 0
        fail_count = 0

        for img_file in image_files:
            img_path = os.path.join(raw_person_dir, img_file)
            result = detect_and_crop_face(img_path, OUTPUT_SIZE)

            if result is not None:
                output_path = os.path.join(processed_person_dir, img_file)
                # 如果原格式不是 jpg，统一存为 jpg
                if os.path.splitext(img_file)[1].lower() != '.jpg':
                    output_path = os.path.splitext(output_path)[0] + '.jpg'
                cv2.imwrite(output_path, result)
                success_count += 1
                print(f"  [成功] {img_file} -> {os.path.basename(output_path)}")
            else:
                fail_count += 1

        total_success += success_count
        total_fail += fail_count
        print(f"  => {person_name}: 成功 {success_count} 张，失败 {fail_count} 张")

    print("\n" + "=" * 60)
    print(f"预处理完成！")
    print(f"  总计成功: {total_success} 张")
    print(f"  总计失败: {total_fail} 张")
    print(f"  输出目录: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()


# """
# 强化人脸预处理脚本
# 功能：
#   - 人脸检测 (Haar Cascade)
#   - 眼睛定位 + 对齐（旋转至双眼水平）
#   - 统一裁剪并缩放至 64×64
#   - CLAHE 光照归一化
#   - 无人脸时使用原图缩放
# 输入目录：raw_photos/  (与脚本同目录)
# 输出目录：processed/   (与脚本同目录)
# """

# import cv2
# import os
# import numpy as np

# # ==================== 配置 ====================
# RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_photos")
# PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
# OUTPUT_SIZE = (64, 64)                     # 最终输出尺寸
# FACE_MARGIN_RATIO = 0.3                    # 裁剪时在检测框外扩的比例
# DESIRED_FACE_HEIGHT = 48                   # 对齐后目标人脸区域高度（用于提取固定区域）

# # 级联分类器路径（OpenCV 自带）
# FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"

# # ==================== 核心函数 ====================

# def align_face(gray, face_rect):
#     """
#     基于眼睛位置旋转人脸使其双眼水平。
#     输入：灰度图，人脸矩形 (x,y,w,h)
#     输出：旋转并裁剪后的脸部区域（未缩放），若无法检测眼睛则返回裁剪后的原始人脸。
#     """
#     x, y, w, h = face_rect
#     # 在人脸上半部分检测眼睛（提高速度）
#     face_roi = gray[y:y+h, x:x+w]
#     eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
#     eyes = eye_cascade.detectMultiScale(
#         face_roi,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(15, 15),
#         maxSize=(w//2, h//3)
#     )

#     if len(eyes) >= 2:
#         # 取最大两只眼睛（按面积排序）
#         eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
#         # 计算眼睛中心（相对 face_roi 坐标）
#         eye_centers = []
#         for (ex, ey, ew, eh) in eyes:
#             cx = ex + ew // 2
#             cy = ey + eh // 2
#             eye_centers.append((cx, cy))
#         # 计算双眼角度
#         (cx1, cy1), (cx2, cy2) = eye_centers
#         if cx1 > cx2:   # 确保左眼在右眼左边
#             (cx1, cy1), (cx2, cy2) = (cx2, cy2), (cx1, cy1)
#         dx = cx2 - cx1
#         dy = cy2 - cy1
#         angle = np.degrees(np.arctan2(dy, dx))   # 需要旋转的角度（度）
#         # 双眼中心在原图中的坐标
#         eye_center_x = int(x + (cx1 + cx2) / 2)
#         eye_center_y = int(y + (cy1 + cy2) / 2)
#         eye_dist = np.sqrt(dx**2 + dy**2)

#         # 旋转整张图片使双眼水平
#         M = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), angle, 1.0)
#         rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
#                                  flags=cv2.INTER_CUBIC)
#         # 根据眼距计算裁剪区域（正方形，中心为旋转后的眼睛中点）
#         desired_face_size = int(eye_dist * 1.8)   # 根据眼距动态调整
#         half = desired_face_size // 2
#         # 裁剪范围不能超出图像
#         x_start = max(0, eye_center_x - half)
#         x_end   = min(rotated.shape[1], eye_center_x + half)
#         y_start = max(0, eye_center_y - half)
#         y_end   = min(rotated.shape[0], eye_center_y + half)
#         aligned = rotated[y_start:y_end, x_start:x_end]
#         return aligned
#     else:
#         # 无法检测眼睛，直接返回原始人脸框（带边距）
#         margin = int(max(w, h) * FACE_MARGIN_RATIO)
#         x1 = max(0, x - margin)
#         y1 = max(0, y - margin)
#         x2 = min(gray.shape[1], x + w + margin)
#         y2 = min(gray.shape[0], y + h + margin)
#         return gray[y1:y2, x1:x2]


# def preprocess_image(image_path):
#     """
#     对单张图像执行完整预处理，返回 64×64 归一化人脸图像。
#     若无人脸，返回整图缩放结果。
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         return None

#     # 转灰度
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img

#     # 人脸检测
#     face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )

#     if len(faces) == 0:
#         # 无人脸：整图缩放 + CLAHE
#         resized = cv2.resize(gray, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         resized = clahe.apply(resized)
#         return resized

#     # 取最大人脸（面积最大）
#     if len(faces) > 1:
#         faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
#     face_rect = faces[0]

#     # 对齐人脸（基于眼睛）
#     aligned_face = align_face(gray, face_rect)

#     # 缩放至统一尺寸
#     if aligned_face is None or aligned_face.size == 0:
#         # 极端情况，回退到简单裁剪
#         x, y, w, h = face_rect
#         margin = int(max(w, h) * FACE_MARGIN_RATIO)
#         x1 = max(0, x - margin)
#         y1 = max(0, y - margin)
#         x2 = min(gray.shape[1], x + w + margin)
#         y2 = min(gray.shape[0], y + h + margin)
#         aligned_face = gray[y1:y2, x1:x2]
#     resized = cv2.resize(aligned_face, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

#     # 光照归一化（CLAHE）
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     resized = clahe.apply(resized)

#     return resized


# def main():
#     print("=" * 60)
#     print("强化人脸预处理 (检测 + 对齐 + 64×64 + CLAHE)")
#     print("=" * 60)

#     os.makedirs(PROCESSED_DIR, exist_ok=True)

#     # 获取所有人物文件夹（支持多种命名格式）
#     person_folders = sorted([
#         d for d in os.listdir(RAW_DIR)
#         if os.path.isdir(os.path.join(RAW_DIR, d))
#     ])

#     if not person_folders:
#         print(f"\n[错误] 在 {RAW_DIR} 中没有找到任何人物文件夹！")
#         print("请在 raw_photos/ 下创建 Person_01 ~ Person_20 并放入照片。")
#         return

#     print(f"找到 {len(person_folders)} 个类别：{person_folders}")

#     total_success, total_fail = 0, 0
#     valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

#     for person_name in person_folders:
#         raw_person_dir = os.path.join(RAW_DIR, person_name)
#         processed_person_dir = os.path.join(PROCESSED_DIR, person_name)
#         os.makedirs(processed_person_dir, exist_ok=True)

#         image_files = sorted([
#             f for f in os.listdir(raw_person_dir)
#             if os.path.splitext(f)[1].lower() in valid_extensions
#         ])
#         if not image_files:
#             print(f"  [警告] {person_name} 文件夹中没有图片")
#             continue

#         print(f"\n--- 处理 {person_name} ({len(image_files)} 张) ---")
#         success, fail = 0, 0

#         for img_file in image_files:
#             img_path = os.path.join(raw_person_dir, img_file)
#             processed_img = preprocess_image(img_path)

#             if processed_img is not None:
#                 # 统一保存为 .jpg
#                 base_name = os.path.splitext(img_file)[0]
#                 out_name = base_name + '.jpg'
#                 out_path = os.path.join(processed_person_dir, out_name)
#                 cv2.imwrite(out_path, processed_img)
#                 success += 1
#                 print(f"  [OK] {img_file} -> {out_name}")
#             else:
#                 fail += 1
#                 print(f"  [失败] 无法读取 {img_file}")

#         total_success += success
#         total_fail += fail
#         print(f"  => 成功 {success} / 失败 {fail}")

#     print("\n" + "=" * 60)
#     print(f"预处理全部完成！")
#     print(f"  总成功: {total_success} 张")
#     print(f"  总失败: {total_fail} 张")
#     print(f"  输出目录: {PROCESSED_DIR}")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()