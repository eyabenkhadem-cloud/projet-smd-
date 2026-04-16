
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


image = cv2.imread(r"C:\Users\ASUS\Desktop\projet smd\im2.jpeg.jfif", cv2.IMREAD_GRAYSCALE)
print("Image shape:", image.shape)
h, w = image.shape

def apply_dct(image):
    image_float = image.astype(np.float32)
    dct_image = np.zeros_like(image_float)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image_float[i:i+8, j:j+8]
            dct_image[i:i+8, j:j+8] = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return dct_image
def calculate_psnr(original, watermarked):
    mse = np.mean((original.astype(np.float32) - watermarked.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(255**2 / mse)

def apply_idct(dct_image):
    image_back = np.zeros_like(dct_image)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_image[i:i+8, j:j+8]
            image_back[i:i+8, j:j+8] = idct(idct(block.T, norm='ortho').T, norm='ortho')
    return np.clip(image_back, 0, 255).astype(np.uint8)

def embed_watermark(dct_image, watermark, delta=25):
    dct_copy = dct_image.copy()
    k = 0
    for i in range(1, dct_image.shape[0], 8):
        for j in range(1, dct_image.shape[1], 8):
            if k >= len(watermark):
                break
            coef = dct_copy[i, j]
            if watermark[k] == 0: 
                dct_copy[i, j] = delta * np.round(coef / delta)   
            else:
                dct_copy[i, j] = delta * (np.round(coef / delta) + 0.5)
            k += 1
    return dct_copy

def extract_watermark(dct_image, watermark_size, delta=25):
    extracted = []
    for i in range(1, dct_image.shape[0], 8):
        for j in range(1, dct_image.shape[1], 8):
            if len(extracted) >= watermark_size:
                break
            coef = dct_image[i, j]
            bit = 1 if (coef / delta) % 1 > 0.25 else 0
            extracted.append(bit)
    return np.array(extracted)

def attack_noise(image):
    noise = np.random.randn(*image.shape) * 10
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def attack_jpeg(image, quality=50):
    return cv2.imdecode(cv2.imencode('.jpg', image,
           [cv2.IMWRITE_JPEG_QUALITY, quality])[1], 0)
def calculate_ber(original_wm, extracted_wm):
    errors = np.sum(original_wm != extracted_wm)
    return errors / len(original_wm)

watermark = np.random.randint(0, 2, (h//8) * (w//8))


dct_image = apply_dct(image)
dct_watermarked = embed_watermark(dct_image, watermark)

watermarked_image = apply_idct(dct_watermarked)
psnr_value = calculate_psnr(image, watermarked_image) 
print(f"PSNR: {psnr_value:.2f} dB")

attacked_noise = attack_noise(watermarked_image)
attacked_jpeg = attack_jpeg(watermarked_image)
wm_size = len(watermark)
extracted_noise = extract_watermark(apply_dct(attacked_noise), wm_size)
extracted_jpeg = extract_watermark(apply_dct(attacked_jpeg), wm_size)

ber_noise = calculate_ber(watermark, extracted_noise)
ber_jpeg = calculate_ber(watermark, extracted_jpeg)
print(f"BER apres bruit: {ber_noise:.4f}")
print(f"BER apres JPEG: {ber_jpeg:.4f}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Image originale")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(watermarked_image, cmap='gray')
plt.title("Image tatouee")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(attacked_noise, cmap='gray')
plt.title("Image attaquee")
plt.axis("off")
plt.tight_layout()
plt.show()