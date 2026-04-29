import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from flask import Flask

app = Flask(__name__)

def apply_dct(image):
    h, w = image.shape
    image_float = image.astype(np.float32)
    dct_image = np.zeros_like(image_float)
    for i in range(0, h - h % 8, 8):
        for j in range(0, w - w % 8, 8):
            block = image_float[i:i+8, j:j+8]
            dct_image[i:i+8, j:j+8] = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return dct_image

def apply_idct(dct_image):
    h, w = dct_image.shape
    image_back = np.zeros_like(dct_image)
    for i in range(0, h - h % 8, 8):
        for j in range(0, w - w % 8, 8):
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
    _, encimg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encimg, 0)

def calculate_psnr(original, watermarked):
    mse = np.mean((original.astype(np.float32) - watermarked.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(255**2 / mse)

def calculate_ber(original_wm, extracted_wm):
    return np.sum(original_wm != extracted_wm) / len(original_wm)


@app.route('/')
def home():
    image = cv2.imread("im2.jpeg.jfif", cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "Error: image not found", 500

    h, w = image.shape
    watermark = np.random.randint(0, 2, (h // 8) * (w // 8))

    dct_image = apply_dct(image)
    dct_watermarked = embed_watermark(dct_image, watermark)
    watermarked_image = apply_idct(dct_watermarked)

    psnr_value = calculate_psnr(image, watermarked_image)

    attacked_noise = attack_noise(watermarked_image)
    attacked_jpeg = attack_jpeg(watermarked_image)

    ber_noise = calculate_ber(watermark, extract_watermark(apply_dct(attacked_noise), len(watermark)))
    ber_jpeg = calculate_ber(watermark, extract_watermark(apply_dct(attacked_jpeg), len(watermark)))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(watermarked_image, cmap='gray')
    plt.title("Watermarked")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(attacked_noise, cmap='gray')
    plt.title("Noisy Attack")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("result.png")

    return f"""
    <h2>Watermark Results</h2>
    <p>PSNR: {psnr_value:.2f} dB</p>
    <p>BER after noise: {ber_noise:.4f}</p>
    <p>BER after JPEG: {ber_jpeg:.4f}</p>
    <img src="/result" style="max-width:100%">
    """

@app.route('/result')
def result():
    return app.send_static_file('../result.png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)