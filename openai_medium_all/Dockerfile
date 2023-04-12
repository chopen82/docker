#FROM python:3.10
FROM chopen82/openai:medium
#COPY requirements.txt .

##RUN pip install -r requirements.txt
#RUN pip install flask
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#RUN apt update && apt install ffmpeg -y
#RUN pip install -U openai-whisper
#RUN python -c "import whisper;modelname = 'large';model = whisper.load_model(modelname)"
#RUN pip install git+https://github.com/openai/whisper.git 
#RUN pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
#RUN pip install pydub

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]