
FROM python:3.11-slim


WORKDIR /app

COPY *.whl ./

RUN pip install --no-cache-dir *.whl 

COPY im2.jpeg.jfif . 

COPY . .

CMD ["python", "watermarkproject.py"]