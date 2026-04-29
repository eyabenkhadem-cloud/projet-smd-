
FROM python:3.11-slim


WORKDIR /app

COPY linux_wheels/ ./linux_wheels/

RUN pip install --no-cache-dir ./linux_wheels/*.whl

COPY im2.jpeg.jfif . 
RUN pip install flask matplotlib
COPY . .

CMD ["python", "watermarkproject.py"]