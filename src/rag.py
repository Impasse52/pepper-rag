import os

import pandas as pd
import torch
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import (
    HuggingFaceLocalGenerator,
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from transformers import pipeline

from .util import preprocess


def setup_documents(df):
    df.columns = ["title", "addr", "text"]
    df = preprocess(df)

    from haystack.dataclasses import Document

    # Population of the Document dataclasses. Title and URLs are added as metadata
    titles = list(df["title"].values)
    texts = list(df["text"].values)
    urls = list(df["addr"].values)

    raw_docs = []
    for title, text, url in zip(titles, texts, urls):
        raw_docs.append(Document(content=text, meta={"name": title or "", "url": url}))

    return raw_docs


def setup_document_store(raw_docs: list, ds_dir: str) -> InMemoryDocumentStore:
    print("Setting up Document Store")

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    if os.path.exists(ds_dir):
        print("Loading document store from disk")
        document_store = document_store.load_from_disk(ds_dir)
    else:
        indexing_pl = Pipeline()

        # preprocess documents by removing common junk
        indexing_pl.add_component("cleaner", DocumentCleaner())

        # split the document every two sentences
        indexing_pl.add_component(
            "splitter", DocumentSplitter(split_by="sentence", split_length=2)
        )

        # compute word embeddings on the documents
        indexing_pl.add_component(
            "doc_embedder",
            SentenceTransformersDocumentEmbedder(
                model="thenlper/gte-large",  # This is the model used
                device=ComponentDevice.from_str(
                    "cuda:0"
                ),  # We use the GPU for this operation
                meta_fields_to_embed=["title"],  # We add the
            ),
        )

        # write documents in document store
        indexing_pl.add_component(
            "writer",
            DocumentWriter(
                document_store=document_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        )

        # Cleaner -> Splitter -> Document Embedder -> Writer
        indexing_pl.connect("cleaner", "splitter")
        indexing_pl.connect("splitter", "doc_embedder")
        indexing_pl.connect("doc_embedder", "writer")

        # run pipeline and cache document store
        indexing_pl.run({"cleaner": {"documents": raw_docs}})
        document_store.save_to_disk(ds_dir)

    return document_store


def setup_llm():
    print("Setting up LLM")
    llm = HuggingFaceLocalGenerator(
        # "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        huggingface_pipeline_kwargs={
            "device_map": "auto",
            "model_kwargs": {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
        },
        generation_kwargs={"max_new_tokens": 100},
    )

    # init the generator
    llm.warm_up()

    return llm


def setup_prompt_builder():
    print("Setting up Prompt Builder")
    prompt_template = """<|system|>Using the information contained in the context, give a comprehensive answer to the question.
    If the answer is contained in the context, also report the source URL.
    If the answer cannot be deduced from the context, do not give an answer.</s>
    <|user|>
    Context:
    {% for doc in documents %}
    {{ doc.content }} URL:{{ doc.meta['url'] }}
    {% endfor %};
    Question: {{query}}
    </s>
    <|assistant|>
    """

    # Since an PromptBuilder object can be associated with only one Pipeline, two copies of the same object had to be made
    prompt_builder = PromptBuilder(template=prompt_template)

    return prompt_builder


def setup_machine_translation():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")


def setup_pipeline(document_store, llm, prompt_builder):
    print("Setting up pipeline")
    pipeline = Pipeline()

    # add each component to the pipeline:
    # Text Embedder -> Retriever -> Prompt Builder -> LLM
    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(
            model="thenlper/gte-large",
            device=ComponentDevice.from_str("cuda:0"),
        ),
    )
    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(
            document_store=document_store,
            top_k=5,
        ),
    )
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("text_embedder", "retriever")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")

    return pipeline


def query_rag(query: str, pl: Pipeline) -> str:
    print("Querying...")
    results = pl.run(
        {
            "text_embedder": {"text": query},
            "prompt_builder": {"query": query},
        }
    )

    output = results["llm"]["replies"][0]

    return output


if __name__ == "__main__":
    filepath = "./data/unipa_dataset.json"

    df = pd.read_json(filepath)
    raw_docs = setup_documents(df)

    ds = setup_document_store(raw_docs, "./output/document_store.json")
    llm = setup_llm()
    pb = setup_prompt_builder()
    mt = setup_machine_translation()

    pl = setup_pipeline(ds, llm, pb)

    output = query_rag("Tell me about Data, Algorithms and Machine Intelligence", pl)

    print(mt(output)[0]["translation_text"])
