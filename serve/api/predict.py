import sys
from . import app
sys.path.append(str(app.config['LIB_PATH']))

from musicautobot.music_transformer import *
from musicautobot.config import *

from flask import Response, send_from_directory, send_file, request, jsonify
from .save import to_s3

import torch
import traceback
torch.set_num_threads(4)

data = load_data(app.config['DATA_PATH'], app.config['DATA_SAVE_NAME'], num_workers=1)
learn = music_model_learner(data, pretrained_path=app.config['MUSIC_MODEL_PATH'])

if torch.cuda.is_available(): learn.model.cuda()
# learn.to_fp16(loss_scale=512) # fp16 not supported for cpu - https://github.com/pytorch/pytorch/issues/17699

@app.route('/predict/midi', methods=['POST'])
def predict_midi():
    args = request.form.to_dict()
    midi = request.files['midi'].read()
    print('THE ARGS PASSED:', args)
    bpm = float(args['bpm']) # (AS) TODO: get bpm from midi file instead
    temperatures = (float(args.get('noteTemp', 1.2)), float(args.get('durationTemp', 0.8)))
    n_words = int(args.get('nSteps', 200))
    seed_len = int(args.get('seedLen', 12))
    # debugging 1 - send exact midi back
    # with open('/tmp/test.mid', 'wb') as f:
    #     f.write(midi)
    # return send_from_directory('/tmp', 'test.mid', mimetype='audio/midi')

    # debugging 2 - test music21 conversion
    # stream = file2stream(midi) # 1.

    # debugging 3 - test npenc conversion
    # seed_np = midi2npenc(midi) # music21 can handle bytes directly 
    # stream = npenc2stream(seed_np, bpm=bpm)

    # debugging 4 - midi in, convert, midi out
    # stream = file2stream(midi) # 1.
    # midi_in = Path(stream.write("musicxml"))
    # print('Midi in:', midi_in)
    # stream_sep = separate_melody_chord(stream)
    # midi_out = Path(stream_sep.write("midi"))
    # print('Midi out:', midi_out)
    # s3_id = to_s3(midi_out, args)
    # result = {
    #     'result': s3_id
    # }
    # return jsonify(result)

    # Main logic
    try:
        full = predict_from_midi(learn, midi=midi, n_words=n_words, seed_len=seed_len, temperatures=temperatures)
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

# @app.route('/midi/song/<path:sid>')
# def get_song_midi(sid):
#     return send_from_directory(file_path/data_dir, htlist[sid]['midi'], mimetype='audio/midi')

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

