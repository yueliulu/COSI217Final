FROM python:3.11-slim
WORKDIR /app
COPY . /app
ENV PYTHONPATH="/app"
RUN pip install -r requirements.txt
EXPOSE 8501
ENV NAME COSI217
CMD ["streamlit", "run", "app/app_streamlit.py"]