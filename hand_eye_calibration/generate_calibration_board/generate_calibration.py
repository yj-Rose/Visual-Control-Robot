#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np

# ===================== 参数 =====================

# ChArUco 棋盘参数（真实物理尺寸，单位：米）
squares_x = 5
squares_y = 7
square_length = 0.04   # 4 cm
marker_length = 0.03   # 3 cm

# ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# 打印分辨率（非常重要）
DPI = 300   # 常用：300（激光 / 喷墨都 OK）

# 边距（物理尺寸，米）
margin_m = 0.01   # 1 cm 白边，方便贴板

# ===================== 单位换算 =====================

METER_TO_INCH = 39.3701

square_px = int(square_length * METER_TO_INCH * DPI)
marker_px = int(marker_length * METER_TO_INCH * DPI)
margin_px = int(margin_m * METER_TO_INCH * DPI)

# 整张图像的像素尺寸（1:1）
img_width_px = squares_x * square_px + 2 * margin_px
img_height_px = squares_y * square_px + 2 * margin_px

print("=== ChArUco Board Physical Spec ===")
print(f"square_length  : {square_length*1000:.1f} mm")
print(f"marker_length  : {marker_length*1000:.1f} mm")
print(f"DPI            : {DPI}")
print(f"image size     : {img_width_px} x {img_height_px} px")

# ===================== 创建 ChArUco Board =====================

board = aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    aruco_dict
)

# ===================== 绘制 1:1 图像 =====================

img = aruco.drawPlanarBoard(
    board,
    (img_width_px, img_height_px),
    marginSize=margin_px,
    borderBits=1
)

# ===================== 保存 PNG =====================

out_png = "charuco_board_1to1.png"
cv2.imwrite(out_png, img)
print(f"Saved: {out_png}")

# ===================== 保存带 DPI 的 PDF（可选） =====================

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader

    pdf_w_in = img_width_px / DPI
    pdf_h_in = img_height_px / DPI

    c = canvas.Canvas("charuco_board_1to1.pdf",
                      pagesize=(pdf_w_in * inch, pdf_h_in * inch))

    img_reader = ImageReader(out_png)
    c.drawImage(img_reader, 0, 0,
                width=pdf_w_in * inch,
                height=pdf_h_in * inch)

    c.showPage()
    c.save()
    print("Saved: charuco_board_1to1.pdf")

except ImportError:
    print("reportlab 未安装，仅生成 PNG")
