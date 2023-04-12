FROM chopen82/translate:en-pl

#COPY requirements.txt .

#RUN pip install transformers
#RUN pip install flask
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#RUN pip install --user -U nltk
#RUN pip install sentencepiece
#RUN pip install sacremoses

#RUN python -c "import nltk; nltk.download('punkt')"
#RUN python -c "from transformers import MarianMTModel,MarianTokenizer;mname = 'gsarti/opus-mt-tc-en-pl';tokenizer = MarianTokenizer.from_pretrained(mname);model = MarianMTModel.from_pretrained(mname)"



COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]