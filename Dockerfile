FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["streamlit", "run", "--server.port", "8080", "--server.address", "0.0.0.0", "streamlit_app.py"]
