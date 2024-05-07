FROM python:3.11-slim
WORKDIR /
COPY ./app /app
COPY ./model /model
COPY ./Dockerfile /
COPY ./requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
ENV NAME COSI217
CMD ["streamlit", "run", "app/app_streamlit.py", "--server.enableCORS=false"]