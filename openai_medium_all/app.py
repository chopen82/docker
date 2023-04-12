from flask import Flask, jsonify, request
import torch
import time
import whisper
import json
import os
from pydub import AudioSegment

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
device
# You can choose your model from - see it on readme file and update the modelname
#modelname = "large-v2"
modelname = "medium"
model = whisper.load_model(modelname)



app = Flask(__name__)

@app.post('/upload')
def predict():
    uploaded_file = request.files['fileName']
    #lang = request.files['language']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        audio_file_path = uploaded_file.filename
        audio_file_path = uploaded_file.filename
        audio = AudioSegment.from_file(audio_file_path)
        start_time = time.time()
        #options = dict(language="pl")
        #transcribe_options = dict(task="transcribe",**options)
        transcribe_options = dict(task="transcribe")
        result = model.transcribe(audio_file_path,fp16=False,**transcribe_options)
        end_time = time.time()
        print(result["text"])
        print("--- %s seconds ---" % (end_time - start_time))
        data = {}
        data['filename'] = uploaded_file.filename
        data['time'] = "--- %s seconds ---" % (end_time - start_time)
        data['content'] = result["text"]
        json_data = json.dumps(data)
        os.remove(uploaded_file.filename)
        print(result)
    return jsonify(
        filename = uploaded_file.filename,
        transcribe_duration = float("%.2f" % (end_time - start_time)),
        device = device,
        content = result["text"],
        language = result["language"],
        duration = audio.duration_seconds,
        sample_rate = audio.frame_rate,
        channels = audio.channels,
        sample_width = audio.sample_width,
        frame_count = audio.frame_count(),
        frame_rate = audio.frame_rate,
        frame_width = audio.frame_width

    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)