#FROM python:3.10
FROM chopen82/spacy:en

#COPY requirements.txt .

#RUN pip install transformers
#RUN pip install flask
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#RUN pip install --user -U nltk
#RUN pip install sentencepiece
#RUN pip install sacremoses

#RUN python -c "import nltk; nltk.download('punkt')"
#RUN python -c "from transformers import MarianMTModel,MarianTokenizer;mname = 'Helsinki-NLP/opus-mt-ru-en';tokenizer = MarianTokenizer.from_pretrained(mname);model = MarianMTModel.from_pretrained(mname)"

#RUN pip install -U pip setuptools wheel
#RUN pip install -U 'spacy[cuda-121]'
#RUN python -c "import spacy"
#RUN python -m spacy download en_core_web_trf



COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]