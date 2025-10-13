import os
import cv2
import uuid
import shutil
from zipfile import ZipFile
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, send_file
from pylibdmtx import pylibdmtx

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'uploads'
USER_OUTPUTS = 'user_outputs'  # базовая папка для всех пользовательских выходов
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(USER_OUTPUTS, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_folder_name(name):
    # Удаляем опасные символы, оставляем только буквы, цифры, подчёркивания и дефисы
    return ''.join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip().replace(' ', '_')[:50]

def extract_datamatrix_regions(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded_objects = pylibdmtx.decode(gray, max_count=100)

    saved_files = []
    for i, obj in enumerate(decoded_objects):
        x, y, w, h = obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        roi = img[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        filename = f"datamatrix_{i+1:03d}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, roi)
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
        if not folder_name:
            folder_name = 'default_output'

        folder_name = sanitize_folder_name(folder_name)
        if not folder_name:
            folder_name = 'unnamed_output'

        output_subdir = os.path.join(USER_OUTPUTS, folder_name)
        os.makedirs(output_subdir, exist_ok=True)

        if file.filename == '':
            flash('Файл не выбран')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                saved_files = extract_datamatrix_regions(filepath, output_subdir)
                if not saved_files:
                    flash('DataMatrix-коды не найдены.')
                    return redirect(request.url)
                else:
                    flash(f'Успешно найдено и сохранено {len(saved_files)} DataMatrix-кодов в папку: "{folder_name}"')
                    return render_template('result.html', 
                                         files=saved_files, 
                                         folder_name=folder_name,
                                         total=len(saved_files))
            except Exception as e:
                flash(f'Ошибка обработки: {str(e)}')
                return redirect(request.url)
        else:
            flash('Недопустимый формат файла. Поддерживаемые: JPG, JPEG, PNG.')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/download_zip/<folder_name>')
def download_zip(folder_name):
    folder_path = os.path.join(USER_OUTPUTS, folder_name)
    if not os.path.isdir(folder_path):
        flash('Папка не найдена.')
        return redirect(url_for('index'))

    # Создаём временный ZIP-архив
    zip_filename = f"{folder_name}.zip"
    zip_path = os.path.join(USER_OUTPUTS, zip_filename)

    with ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.jpg'):
                    zipf.write(os.path.join(root, file), arcname=file)

    return send_file(zip_path, as_attachment=True)

@app.route('/outputs/<folder_name>/<filename>')
def download_file(folder_name, filename):
    folder_path = os.path.join(USER_OUTPUTS, folder_name)
    return send_from_directory(folder_path, filename)

if __name__ == '__main__':
    app.run(debug=True)