set up venv

run `pip install -r requirements.txt`

run `wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip`

run `unzip vosk-model-en-us-0.22.zip`

run `python3 -c "import openwakeword; openwakeword.utils.download_models()"`