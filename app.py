from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from PIL import Image
import os
import io
from datetime import datetime
from os.path import basename  # Импортируем basename для фильтра

app = Flask(__name__)

# Папка для сохранения выровненных QR-кодов
OUTPUT_DIR = "output_qr_codes"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Добавляем Jinja2-фильтр для извлечения имени файла
app.jinja_env.filters['basename'] = basename  # <--- ФИЛЬТР ДОБАВЛЕН ЗДЕСЬ

def process_image(file_path):
    """Обрабатывает изображение, находит и выравнивает QR-коды."""
    # Читаем изображение
    image = cv2.imread(file_path)
    if image is None:
        return None, "Ошибка: не удалось загрузить изображение."

    # Конвертируем в оттенки серого для декодирования
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Находим QR-коды
    detector = cv2.QRCodeDetector()
    ok, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(gray)

    if not ok:
        return None, "QR-коды не найдены."

    output_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Обрабатываем каждый QR-код
    if straight_qrcode is not None:
        for i, qr_image in enumerate(straight_qrcode):
            # Сохраняем выровненный QR-код
            output_path = os.path.join(OUTPUT_DIR, f"qr_code_{timestamp}_{i}.jpg")
            
            # Преобразуем изображение в 8-битное целое число без знака
            qr_image_uint8 = np.uint8(qr_image)
            
            # Сохраняем изображение
            cv2.imwrite(output_path, qr_image_uint8)
            output_files.append(output_path)

    return output_files, None

@app.route('/')
def index():
    """Главная страница с формой загрузки."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обрабатывает загрузку файла и возвращает результаты."""
    if 'file' not in request.files:
        return render_template('index.html', error="Файл не выбран.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="Файл не выбран.")

    if file and file.filename.lower().endswith('.jpg'):
        # Сохраняем загруженный файл
        upload_path = os.path.join('uploads', file.filename)
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file.save(upload_path)

        # Обрабатываем изображение
        output_files, error = process_image(upload_path)
        
        if error:
            return render_template('index.html', error=error)
        
        return render_template('result.html', files=output_files)
    
    return render_template('index.html', error="Поддерживаются только файлы JPG.")

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание обработанного QR-кода."""
    return send_file(os.path.join(OUTPUT_DIR, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)