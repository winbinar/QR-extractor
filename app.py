import os
import cv2
import numpy as np
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, send_file
from zipfile import ZipFile

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'uploads'
USER_OUTPUTS = 'user_outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_OUTPUTS, exist_ok=True)

def sanitize_folder_name(name):
    return ''.join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip().replace(' ', '_')[:50] or 'output'

def extract_all_codes(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")

    all_results = []  # [(points, type_name), ...]

    # === 1. Используем BarcodeDetector (поддерживает QR, DataMatrix, линейные штрихкоды) ===
    try:
        detector = cv2.barcode_BarcodeDetector()
        result = detector.detectAndDecode(img)

        if len(result) == 4:
            retval, decoded_info, decoded_type, corners = result
        elif len(result) == 3:
            retval, decoded_info, corners = result
            decoded_type = None
        else:
            retval, corners = False, None

        if retval and corners is not None:
            for i in range(len(corners)):
                pts = corners[i]
                if pts is None or len(pts) < 2:
                    continue

                # Определяем тип
                code_type = "barcode"  # по умолчанию — линейный
                if decoded_type is not None and i < len(decoded_type):
                    t = decoded_type[i]
                    if t == 16:
                        code_type = "datamatrix"
                    elif t in (1, 2, 3, 4):  # QR
                        code_type = "qr"
                    # Иначе — остаётся "barcode"
                else:
                    # Эвристика: если 4 угла → 2D-код (QR или DataMatrix), иначе — линейный
                    if len(pts) == 4:
                        code_type = "qr_or_datamatrix"  # уточним позже
                    else:
                        code_type = "barcode"

                all_results.append((pts, code_type))
    except Exception as e:
        print(f"BarcodeDetector error: {e}")
        pass  # продолжаем, попробуем другие методы

    # === 2. Дополнительно: QRCodeDetector (на случай, если BarcodeDetector пропустил) ===
    try:
        qr_detector = cv2.QRCodeDetector()
        data, points, _ = qr_detector.detectAndDecode(img)
        if data and points is not None and len(points) == 4:
            # Проверим, не добавлен ли уже
            already_found = any(
                len(pts) == 4 and np.allclose(pts, points.reshape(4, 2), atol=10)
                for pts, _ in all_results
            )
            if not already_found:
                all_results.append((points.reshape(4, 2), "qr"))
    except Exception as e:
        print(f"QRCodeDetector fallback error: {e}")
        pass

    saved_files = []

    for idx, (pts, code_type) in enumerate(all_results):
        pts = np.array(pts, dtype=np.float32)

        # === Обработка 2D-кодов (4 угла) ===
        if len(pts) == 4:
            # Уточняем тип: если неизвестен — оставляем как есть
            if code_type == "qr_or_datamatrix":
                code_type = "code2d"  # нейтральное имя

            # Перспективное выравнивание
            def dist(a, b):
                return np.linalg.norm(np.array(a) - np.array(b))

            try:
                w = int(max(dist(pts[0], pts[1]), dist(pts[2], pts[3])))
                h = int(max(dist(pts[0], pts[3]), dist(pts[1], pts[2])))
            except:
                continue

            if w < 10 or h < 10:
                continue

            dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
            try:
                M = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(img, M, (w, h))
            except cv2.error:
                continue

        # === Обработка линейных штрихкодов (2 точки — начало и конец) ===
        elif len(pts) >= 2:
            # Берём первую и последнюю точку
            p1, p2 = pts[0], pts[-1]
            length = int(np.linalg.norm(np.array(p1) - np.array(p2)))
            if length < 20:
                continue

            # Создаём прямоугольник вокруг линии (ширина = длина, высота = 50)
            w, h = length, 50
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi

            # Поворачиваем изображение, чтобы выровнять штрихкод
            center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M_rot, (img.shape[1], img.shape[0]))

            # Обрезаем регион
            x1 = int(center[0] - w // 2)
            y1 = int(center[1] - h // 2)
            x2 = x1 + w
            y2 = y1 + h

            if x1 < 0 or y1 < 0 or x2 > rotated.shape[1] or y2 > rotated.shape[0]:
                # Если выходит за границы — пропускаем
                continue

            warped = rotated[y1:y2, x1:x2]

        else:
            continue

        # Сохраняем
        filename = f"{code_type}_{idx+1:03d}.jpg"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, warped)
        saved_files.append(filename)

    return saved_files

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Файл не выбран')
            return redirect(request.url)

        file = request.files['file']
        folder_name = request.form.get('folder_name', '').strip()
        folder_name = sanitize_folder_name(folder_name)

        output_dir = os.path.join(USER_OUTPUTS, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        if file.filename == '':
            flash('Файл не выбран')
            return redirect(request.url)

        allowed = {'jpg', 'jpeg', 'png'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed:
            flash('Поддерживаются только JPG и PNG')
            return redirect(request.url)

        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            saved_files = extract_all_codes(filepath, output_dir)
            if not saved_files:
                flash('QR-коды, DataMatrix и штрихкоды не найдены.')
                return redirect(request.url)
            flash(f'Найдено и сохранено {len(saved_files)} кодов в папку: "{folder_name}"')
            return render_template('result.html', files=saved_files, folder_name=folder_name, total=len(saved_files))
        except Exception as e:
            flash(f'Ошибка обработки: {str(e)}')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/download_zip/<folder_name>')
def download_zip(folder_name):
    folder_path = os.path.join(USER_OUTPUTS, folder_name)
    if not os.path.isdir(folder_path):
        flash('Папка не найдена.')
        return redirect(url_for('index'))

    zip_path = os.path.join(USER_OUTPUTS, f"{folder_name}.zip")
    with ZipFile(zip_path, 'w') as zipf:
        for f in os.listdir(folder_path):
            if f.endswith('.jpg'):
                zipf.write(os.path.join(folder_path, f), f)

    return send_file(zip_path, as_attachment=True)

@app.route('/outputs/<folder_name>/<filename>')
def download_file(folder_name, filename):
    return send_from_directory(os.path.join(USER_OUTPUTS, folder_name), filename)

if __name__ == '__main__':
    app.run(debug=True)