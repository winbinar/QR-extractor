# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
# Мы копируем только этот файл, а не весь код, 
# так как код будет монтироваться как том (volume)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем директории, которые использует приложение
RUN mkdir -p /app/uploads && mkdir -p /app/output_qr_codes

# Открываем порт, на котором будет работать Gunicorn
EXPOSE 5000

# Команда для запуска приложения с Gunicorn и автоматической перезагрузкой
# --reload будет следить за изменениями в файлах и перезапускать воркеры
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=2", "--reload", "app:app"]
