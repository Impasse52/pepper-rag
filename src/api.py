# NOTA: è opportuno eseguire questo script solo su Linux poiché
# qi per Python3 è disponibile solo su Linux (WSL va bene).
# L'alternativa sarebbe di fare un downgrade a Python2.7, ma
# non è garantito che tutte le altre dipendenze lo supporteranno.

import time
from io import BytesIO

import pandas as pd
import speech_recognition as sr
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .pepper import init_pepper
from .rag import (
    query_rag,
    setup_document_store,
    setup_documents,
    setup_llm,
    setup_machine_translation,
    setup_pipeline,
    setup_prompt_builder,
)

app = FastAPI()


# --- init connection to Pepper ---
pepper_ip = "192.168.1.13"
pepper_port = 9559
pepper = init_pepper(pepper_ip, pepper_port)

# --- initialize the pipeline, takes some time (is exec only once) ---
start_time = time.time()
filepath = "./data/unipa_dataset.json"

df = pd.read_json(filepath)
raw_docs = setup_documents(df)

load_dotenv()

ds = setup_document_store(raw_docs, "./output/document_store.json")
llm = setup_llm()
pb = setup_prompt_builder()
mt = setup_machine_translation()

pl = setup_pipeline(ds, llm, pb)
print(f"Initialized server in {time.time() - start_time:.2f}s.")


@app.post("/sr/")
async def do_speech_recognition(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(
            status_code=400,
            content={"message": "No file was sent."},
        )

    # todo: replace with logger
    print("#" * 80)
    print(f"Performing speech processing on: {file.filename}")
    print("#" * 80)

    wav_bytes = await file.read()
    audio_file = BytesIO(wav_bytes)

    r = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)

    text = r.recognize_google(audio_data, language="it-IT")

    return {
        "text": text,
        "name": file.filename,
        "file_size": len(wav_bytes),
    }


@app.get("/query")
async def call_query_pipeline(query: str):
    output = query_rag(query, pl)
    translated_out = mt(output)[0]["translation_text"]

    print(translated_out)
    return translated_out


@app.get("/tts")
async def perform_tts(text: str):
    pepper.tts(text)
