from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from pyzbar.pyzbar import decode
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

def align_qr_code(image, qr):
    """Выравнивает QR-код и возвращает выровненное изображение."""
    # Получаем координаты углов QR-кода
    points = qr.polygon
    if len(points) < 4:
        return None

    # Извлекаем углы
    pts = np.array([(p.x, p.y) for p in points], dtype="float32")

    # Определяем размеры QR-кода
    width = int(np.sqrt((pts[0][0] - pts[1][0])**2 + (pts[0][1] - pts[1][1])**2))
    height = int(np.sqrt((pts[1][0] - pts[2][0])**2 + (pts[1][1] - pts[2][1])**2))

    # Целевые координаты для выравнивания
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype="float32")

    # Вычисляем матрицу преобразования
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # Выполняем преобразование
    aligned = cv2.warpPerspective(image, M, (width, height))

    return aligned

def process_image(file_path):
    """Обрабатывает изображение, находит и выравнивает QR-коды."""
    # Читаем изображение
    image = cv2.imread(file_path)
    if image is None:
        return None, "Ошибка: не удалось загрузить изображение."

    # Конвертируем в оттенки серого для декодирования
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Находим QR-коды
    qr_codes = decode(gray)
    if not qr_codes:
        return None, "QR-коды не найдены."

    output_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Обрабатываем каждый QR-код
    for i, qr in enumerate(qr_codes):
        aligned_qr = align_qr_code(image, qr)
        if aligned_qr is None:
            continue

        # Сохраняем выровненный QR-код
        output_path = os.path.join(OUTPUT_DIR, f"qr_code_{timestamp}_{i}.jpg")
        cv2.imwrite(output_path, aligned_qr)
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