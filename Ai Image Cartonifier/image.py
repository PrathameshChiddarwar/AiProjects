import cv2
import numpy as np

# Load image
img = cv2.imread("D:\Project5-5-2026\Ai Image Cartonifier\photo professional.jpg")

if img is None:
    print("Image not found")
    exit()

# Resize
img = cv2.resize(img, (900, 900))

# =========================
# Light Smoothing
# =========================

smooth = cv2.bilateralFilter(img, 7, 50, 50)

# =========================
# Color Quantization
# =========================

data = np.float32(smooth).reshape((-1, 3))

K = 10

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    20,
    1.0
)

_, label, center = cv2.kmeans(
    data,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

center = np.uint8(center)

quantized = center[label.flatten()]
quantized = quantized.reshape(smooth.shape)

# =========================
# Better Edge Detection
# =========================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3,3), 0)

edges = cv2.Canny(gray, 100, 180)

# Thicker edges
kernel = np.ones((2,2), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

# Invert
edges = cv2.bitwise_not(edges)

# =========================
# Combine
# =========================

cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

# =========================
# Sharpen
# =========================

kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])

cartoon = cv2.filter2D(cartoon, -1, kernel)

# =========================
# Contrast Boost
# =========================

cartoon = cv2.convertScaleAbs(cartoon, alpha=1.15, beta=10)

# =========================
# Save
# =========================

cv2.imwrite("sharp_cartoon.jpg", cartoon)

# Display
cv2.imshow("Cartoon", cartoon)

cv2.waitKey(0)
cv2.destroyAllWindows()