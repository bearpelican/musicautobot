import sys, os
from . import app
sys.path.append(str(app.config['LIB_PATH']))

from musicautobot.multitask_transformer import *
from musicautobot.music_transformer import *
from musicautobot.config import *
from flask import Response, send_from_directory, send_file, request, jsonify

from .save import to_s3

import torch
import traceback
torch.set_num_threads(4)

data = load_data(app.config['DATA_PATH'], app.config['DATA_SAVE_NAME'], num_workers=1)
learn = multitask_model_learner(data, pretrained_path=app.config['MULTITASK_MODEL_PATH'])

if torch.cuda.is_available(): learn.model.cuda()


@app.route('/predict/midi', methods=['POST'])
def predict_midi():
    args = request.form.to_dict()
    midi = request.files['midi'].read()
    print('Prediction Args:', args)

    # Universal parameters
    bpm = float(args['bpm']) # (AS) TODO: get bpm from midi file instead
    prediction_type = args.get('predictionType', 'next') 
    temperatures = (float(args.get('noteTemp', 1.2)), float(args.get('durationTemp', 0.8)))
    top_k, top_p = (int(args.get('topK', 20)), float(args.get('topP', 0.9)))

    # Parameters for NextSeq and Melody/Chords
    n_words = int(args.get('nSteps', 200))
    seed_len = int(args.get('seedLen', 12))

    # Parameters for Masking
    mask_start, mask_end = None, None
    try:
        mask_start = int(args['maskStart'])
        mask_end = int(args['maskEnd'])
    except: pass

    # Main logic
    try:
        if prediction_type == 'next':
            full = nw_predict_from_midi(learn, midi=midi, n_words=n_words, seed_len=seed_len, temperatures=temperatures, top_k=top_k, top_p=top_p)
            stream = separate_melody_chord(full.to_stream(bpm=bpm))
        elif prediction_type in ['melody', 'chords']:
            full = s2s_predict_from_midi(learn, midi=midi, n_words=n_words, temperatures=temperatures, seed_len=seed_len, 
                                         pred_melody=(prediction_type == 'melody'), use_memory=True, top_k=top_k, top_p=top_p)
            stream = full.to_stream(bpm=bpm)
        elif prediction_type in ['pitch', 'rhythm']:
            full = mask_predict_from_midi(learn, midi=midi, temperatures=temperatures, predict_notes=(prediction_type == 'pitch'), section=(mask_start, mask_end), top_k=top_k, top_p=top_p)
            stream = separate_melody_chord(full.to_stream(bpm=bpm))
        midi_out = Path(stream.write("midi"))
        print('Wrote to temporary file:', midi_out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to predict: {e}'})

    s3_id = to_s3(midi_out, args)
    result = {
        'result': s3_id
    }
    return jsonify(result)
    # return send_from_directory(midi_out.parent, midi_out.name, mimetype='audio/midi')

@app.route('/midi/convert', methods=['POST'])
def convert_midi():
    args = request.form.to_dict()
    if 'midi' in request.files:
        midi = request.files['midi'].read()
    elif 'midi_path'in args:
        midi = args['midi_path']

    stream = file2stream(midi) # 1.
    # stream = file2stream(midi).chordify() # 1.
    stream_out = Path(stream.write('musicxml'))
    return send_from_directory(stream_out.parent, stream_out.name, mimetype='xml')