import os
import cv2
import numpy as np
import uuid
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, send_file
from zipfile import ZipFile
from pylibdmtx.pylibdmtx import decode as dmtx_decode

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)


UPLOAD_FOLDER = 'uploads'
USER_OUTPUTS = 'user_outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_OUTPUTS, exist_ok=True)

def sanitize_folder_name(name):
    return ''.join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip().replace(' ', '_')[:50] or 'output'

def extract_all_codes(image_path, output_dir):
    app.logger.debug(f"Загружаем изображение: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    all_regions = []

    # === 1. Pylibdmtx — для DataMatrix ===
    try:
        decoded_dmtx = dmtx_decode(gray)
        if decoded_dmtx:
            app.logger.debug(f"Pylibdmtx нашёл {len(decoded_dmtx)} DataMatrix кодов")
            for code in decoded_dmtx:
                # Используем code.polygon, который содержит реальные углы кода,
                # а не просто ограничивающий прямоугольник code.rect
                if hasattr(code, 'polygon') and len(code.polygon) == 4:
                    pts = np.array(code.polygon, dtype=np.float32)
                    all_regions.append((pts, "datamatrix_dmtx"))
                else:
                    # Фоллбэк на случай, если polygon недоступен
                    x, y, w, h = code.rect
                    pts = np.array([
                        [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                    ], dtype=np.float32)
                    all_regions.append((pts, "datamatrix_dmtx_rect"))
    except Exception as e:
        app.logger.error(f"Pylibdmtx error: {e}")


    # === 2. OpenCV BarcodeDetector — для DataMatrix и линейных штрихкодов ===
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
            app.logger.debug(f"OpenCV BarcodeDetector нашёл {len(corners)} кодов")
            for i in range(len(corners)):
                pts = corners[i]
                if pts is None or len(pts) < 2:
                    app.logger.debug(f"Пропускаем код {i+1}: pts is None или меньше 2 точек")
                    continue

                code_type = "barcode"
                if decoded_type is not None and i < len(decoded_type) and decoded_type[i]:
                    t = decoded_type[i]
                    code_type = f"opencv_type_{t}"

                app.logger.debug(f"Найден {code_type} с {len(pts)} точками")
                all_regions.append((pts, code_type))
        else:
            app.logger.debug("OpenCV BarcodeDetector не нашёл кодов")
    except Exception as e:
        app.logger.error(f"OpenCV BarcodeDetector error: {e}")

    # === 3. Контурный поиск — для QR-кодов и других прямоугольных кодов ===
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    app.logger.debug(f"Найдено {len(contours)} контуров")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = cv2.boundingRect(approx)
            w, h = rect[2], rect[3]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            if 0.8 < aspect_ratio < 1.2:
                is_duplicate = False
                for existing_pts, _ in all_regions:
                    if np.linalg.norm(np.mean(existing_pts, axis=0) - np.mean(pts, axis=0)) < 10:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    app.logger.debug(f"Контур {i+1}: найден 4-угольник, площадь={area}, размер={w}x{h}")
                    all_regions.append((pts, "qr_contour"))

    app.logger.info(f"Всего найдено {len(all_regions)} уникальных регионов для обработки")

    saved_files = []
    processed_centers = []

    for idx, (pts, code_type) in enumerate(all_regions):
        pts = np.array(pts, dtype=np.float32)

        center = np.mean(pts, axis=0)
        is_duplicate = False
        for proc_center in processed_centers:
            if np.linalg.norm(center - proc_center) < 15:
                is_duplicate = True
                app.logger.debug(f"Регион {idx+1} ({code_type}) пропущен как дубликат")
                break
        if is_duplicate:
            continue
        
        processed_centers.append(center)
        
        warped = None
        # === Обработка 4-угольных кодов ===
        if len(pts) == 4:
            # === Специальная, надежная обработка для pylibdmtx ===
            if code_type == "datamatrix_dmtx":
                try:
                    # Находим min/max координаты, чтобы получить правильный bounding box
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    x = int(np.min(x_coords))
                    y = int(np.min(y_coords))
                    w_rect = int(np.max(x_coords) - x)
                    h_rect = int(np.max(y_coords) - y)

                    if w_rect > 10 and h_rect > 10:
                        # Простое вырезание области (кроп)
                        warped = img[y:y+h_rect, x:x+w_rect]
                    else:
                        app.logger.debug(f"Регион {idx+1} ({code_type}): слишком маленький ({w_rect}x{h_rect}) — пропускаем")
                        continue
                except Exception as e:
                    app.logger.error(f"Ошибка кропа для {code_type} региона {idx+1}: {e}")
                    continue
            # === Обработка для всех остальных 4-угольных кодов (с исправлением перспективы) ===
            else:
                try:
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    pts = rect

                    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
                    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
                except Exception as e:
                    app.logger.error(f"Ошибка при вычислении размеров для региона {idx+1}: {e}")
                    continue

                if w < 10 or h < 10:
                    app.logger.debug(f"Регион {idx+1}: слишком маленький ({w}x{h}) — пропускаем")
                    continue

                dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
                try:
                    M = cv2.getPerspectiveTransform(pts, dst)
                    warped = cv2.warpPerspective(img, M, (w, h))
                except cv2.error as e:
                    app.logger.error(f"Ошибка перспективного преобразования для региона {idx+1}: {e}")
                    continue
        else:
            app.logger.debug(f"Регион {idx+1} ({code_type}): не 4-угольный, пропускаем")
            continue

        if warped is None:
            app.logger.debug(f"Регион {idx+1} ({code_type}): не удалось извлечь изображение — пропускаем")
            continue

        filename = f"{code_type}_{idx+1:03d}.jpg"
        out_path = os.path.join(output_dir, filename)
        try:
            cv2.imwrite(out_path, warped)
            app.logger.info(f"Сохранён файл: {filename}")
            saved_files.append(filename)
        except Exception as e:
            app.logger.error(f"Ошибка сохранения файла {filename}: {e}")

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
            app.logger.error(f"Ошибка обработки: {e}", exc_info=True)
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
    app.run(debug=True, host='0.0.0.0', port=5001)
