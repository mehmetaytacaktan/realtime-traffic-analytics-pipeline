# 1. Python 3.10 tabanlı hafif Linux imajı
FROM python:3.10-slim

# 2. Sistem bağımlılıklarını ve FFmpeg'i yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt-get/lists/*

# 3. Çalışma dizinini ayarla
WORKDIR /app

# 4. Bağımlılıkları kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Hugging Face Spaces için kullanıcı izinlerini ve dizinlerini ayarla
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# 6. Gradio portunu dışarı aç
EXPOSE 7860

# 7. Uygulamayı başlat
CMD ["python", "app.py"]