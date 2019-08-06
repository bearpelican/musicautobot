# Flask API Endpoint for Music Generation

This API is build specifically for the front end app - musicautobot.com

See: https://github.com/bearpelican/musicautobot_vueapp for the client code

Installation:

*Make sure you have already created musicautobot conda environment*

cd serve
conda env update -f environment.yml

Set S3 BUCKET in api/api.cfg


Running server:

conda activate musicautobot

Local Host:
python run.py

Production:
gunicorn --certfile SSL_CERT --keyfile SSL_KEY -b 127.0.0.1:5000 run_guni:app  --timeout 180 --workers 16