from flask import Flask, jsonify, request
import torch
import spacy
import time
import json
import os
import codecs
from itertools import *

if torch.cuda.is_available():  
  dev = "cuda"
else:  
  dev = "cpu" 
#dev = "cpu"
device = torch.device(dev)

nlp = spacy.load("en_core_web_trf")


app = Flask(__name__)

@app.post('/upload')
def predict():
    data = request.get_json(force=True) 
    try:
        sample = data.get('content_en')
    except KeyError:
        return jsonify({'error': 'No text sent'})

    doc = nlp(sample)
    
    person=[]
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            print(ent.text, ent.label_)
            #person.data.append(ent.text)

    entities = dict()
    for key, g in groupby(sorted(doc.ents, key=lambda x: x.label_), lambda x: x.label_):
        seen = set()
        l = []
        for ent in list(g):
            #if ent.text not in seen:
                seen.add(ent.text)
                l.append(ent.text)
                entities[key] = l
    
    
    
    #print(entities['PERSON'])

    data = {}
    data['content'] = sample
    data['entities'] = entities
    #json_data = json.dumps(data,ensure_ascii=False).encode('utf-8')
    #print(json_data)
    json_data = json.dumps(data, ensure_ascii=False)
    #json_data = doc.to_json()
    
    #print(json_data)
    #with codecs.open("1.json", "w",encoding='utf-8') as outfile:
        # encoded =  json.dumps(json_data,ensure_ascii=False).encode('utf-8')
        # json.dump(json_data, outfile, indent=4,ensure_ascii=False) 



    try:
        result = jsonify(
            content = sample,
            entities = entities
            #content_en = translated_text
        )
    except TypeError as e:
        result = jsonify({'error': str(e)})
    return result

    


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)