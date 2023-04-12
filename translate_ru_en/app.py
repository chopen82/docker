from flask import Flask, jsonify, request
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
from nltk.tokenize import wordpunct_tokenize
import math
import torch
import nltk
import time
import json
import os



if torch.cuda.is_available():  
  dev = "cuda"
else:  
  dev = "cpu" 
#dev = "cpu"
device = torch.device(dev)


#nltk.download('punkt')
mname = 'Helsinki-NLP/opus-mt-ru-en'
tokenizer = MarianTokenizer.from_pretrained(mname)
model = MarianMTModel.from_pretrained(mname)
model.to(device)



app = Flask(__name__)

@app.post('/upload')
def predict():
    data = request.get_json(force=True) 
    try:
        sample = data.get('content')
    except KeyError:
        return jsonify({'error': 'No text sent'})

    #sample = [sample]
    #predictions = predict_pipeline(sample)
    lt = LineTokenizer()
    batch_size = 8

    paragraphs = lt.tokenize(sample)   
    translated_paragraphs = []

    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph, language="russian")
        batches = math.ceil(len(sentences) / batch_size)     
        translated = []
        for i in range(batches):
            sent_batch = sentences[i*batch_size:(i+1)*batch_size]
            model_inputs = tokenizer(sent_batch, return_tensors="pt", padding=True, truncation=True, max_length=500).to(device)
            with torch.no_grad():
                translated_batch = model.generate(**model_inputs)
            translated += translated_batch
        translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translated_paragraphs += [" ".join(translated)]

    translated_text = "\n".join(translated_paragraphs)

    try:
        result = jsonify(
            content_ru = sample,
            content_en = translated_text
        )
    except TypeError as e:
        result = jsonify({'error': str(e)})
    return result

    


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)