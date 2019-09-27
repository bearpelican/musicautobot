
import uuid
import boto3
import json
from pathlib import Path
from . import app
from flask import Response, send_from_directory, send_file, request, jsonify

s3 = boto3.client('s3')
bucket = app.config['S3_BUCKET_NAME']

def to_s3(file, args, base_dir='generated/'):
    s3_id = str(uuid.uuid4()).replace('-', '')
    s3_file = base_dir + s3_id + '.mid'
    s3_json = base_dir + s3_id + '.json'
    
    if not isinstance(file, (str, Path)):
        tmp_midi = '/tmp/' + s3_id + '.mid'
        with open(tmp_midi, 'wb') as f:
            f.write(file)
    else: 
        tmp_midi = file

    if not isinstance(args, (str, Path)):
        tmp_json = '/tmp/' + s3_id + '.json'
        with open(tmp_json, 'w') as f:
            json.dump(args, f)
    else: tmp_json = args
    
    # Uploads the given file using a managed uploader, which will split up large
    # files automatically and upload parts in parallel.
    s3.upload_file(str(tmp_midi), bucket, s3_file)
    s3.upload_file(str(tmp_json), bucket, s3_json)
    print('Saved IDS:', s3_id, s3_id[::-1])
    return s3_id[::-1]

@app.route('/store/save', methods=['POST'])
def save_store():
    args = request.form.to_dict()
    midi = request.files['midi'].read()
    print('Saving store:', args)
#    stream = file2stream(midi) # 1.
#    midi_out = Path(stream.write("midi"))
    s3_id = to_s3(midi, args)
    result = {
        'result': s3_id
    }
    return jsonify(result)
