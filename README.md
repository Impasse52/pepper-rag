# pepper-rag
A simple voice assistant for **Pepper** (SoftBank Robotics) that answers spoken questions in Italian about the University of Palermo (`data/unipa_dataset.json`, obtained through a [custom crawler](https://github.com/pierfrancescomartinello/NLP-Project/blob/main/src/crawler.py)).

The system listens for a question, transcribes it and retrieves the most relevant passages from a knowledge base, which are then used to generate a grounded answer through a local LLM downloaded from HuggingFace. The pipeline is build with Haystack and served with FastAPI.

Requires an NVIDIA GPU with CUDA since the embedder uses `cuda:0` and the LLM is loaded in 4-bit with `bitsandbytes`, a Pepper robot on the local network, and a microphone on the machine that runs the client. The pipeline is built with Haystack and served with FastAPI.

Developed as a proof of concept during a curricular internship at the Italian National Research Council (CNR).

## How it works
1. **Recording** (`src/client.py`, `src/speech_to_text.py`): the client records 5 seconds of audio from the microphone (PyAudio, 44.1 kHz) and uploads the WAV file to the server.
2. **Speech recognition** (`POST /sr`): the server transcribes the audio with the SpeechRecognition library, using Google's web speech API with it-IT.
3. **Retrieval** (`src/rag.py`): the question is embedded with thenlper/gte-large (Sentence Transformers) and the top 5 most similar chunks are retrieved by cosine similarity from a Haystack InMemoryDocumentStore.
4. **Generation** (`GET /query`): the chunks and the question are inserted in a prompt for HuggingFaceH4/zephyr-7b-beta, loaded locally in 4-bit (NF4) quantization. The prompt asks the model to answer only from the context and to report the source URL.
5. **Translation**: the model's answer is in English and is translated to Italian with Helsinki-NLP/opus-mt-en-it.
Speech (`GET /tts`, `src/pepper.py`): the answer is sent to Pepper through the qi SDK and spoken with the robot's ALTextToSpeech service.

---

Note that since this is a quick proof of concept there are a few relevant issues. Several dependencies (namely [qi](https://pypi.org/project/qi/)) rely on Linux. For this reason, this is the suggested operating system to run the project on. On Windows, WSL is supported and must be used in order to launch the server (`src/api.py`). 

Remember to boot into WSL if on Windows (e.g. `wsl -d ubuntu`), then:
```bash
git clone https://github.com/Impasse52/pepper_rag
cd pepper_rag
python3 -m venv pepper_rag
source pepper_rag/bin/activate
pip install -r requirements.txt # will take a long while
```


Copy `.env.example` to `.env` and match your variables to your setup. 
| Variable | Default | Used by | Description |
|---|---|---|---|
| `PEPPER_IP` | required | server | Pepper's IP address |
| `PEPPER_PORT` | `9559` | server | NAOqi port |
| `SR_LANGUAGE` | `it-IT` | server | Speech recognition language |
| `API_URL` | `http://127.0.0.1:8000` | client | Server address |
| `RECORD_SECONDS` | `5` | client | Question recording length |

Then, the server can be launched with `fastapi run src/api.py`. The client (`src/client.py`) must be launched from the host OS (i.e. from Windows directly). This can be done by running `python3 src/client.py`, starting the recording which will then be sent to the server.
