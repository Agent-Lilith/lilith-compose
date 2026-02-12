"""
FastAPI + spaCy NER API.
Supports: en, fr, de, nl, ru (md models); ar, ms use multilingual NER (xx_ent_wiki_sm).
"""

from contextlib import asynccontextmanager

import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Language code -> spaCy model name (md where available; ar/ms use multilingual)
MODELS = {
    "en": "en_core_web_md",
    "fr": "fr_core_news_md",
    "de": "de_core_news_md",
    "nl": "nl_core_news_md",
    "ru": "ru_core_news_md",
    "ar": "xx_ent_wiki_sm",  # no official Arabic NER; use multilingual
    "ms": "xx_ent_wiki_sm",  # no Malaysian; use multilingual
}
SUPPORTED_LANGS = list(MODELS.keys())
# Load with NER only to save memory (disable tagger, parser, lemmatizer, etc.)
DISABLE = [
    "tok2vec",
    "tagger",
    "morphologizer",
    "parser",
    "attribute_ruler",
    "lemmatizer",
    "senter",
]


def load_nlp(lang: str):
    model_name = MODELS[lang]
    try:
        nlp = spacy.load(model_name, exclude=DISABLE)
    except OSError:
        raise RuntimeError(
            f"Model {model_name} not found. Run: python -m spacy download {model_name}"
        )
    return nlp


# In-memory cache: one nlp per language (lazy-loaded)
_nlp_cache: dict[str, spacy.Language] = {}


def get_nlp(lang: str) -> spacy.Language:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language: {lang}. Supported: {SUPPORTED_LANGS}")
    if lang not in _nlp_cache:
        _nlp_cache[lang] = load_nlp(lang)
    return _nlp_cache[lang]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: preload no models to keep startup fast and memory low
    yield
    _nlp_cache.clear()


app = FastAPI(
    title="spaCy NER API",
    description="Named entity recognition for en, fr, de, nl, ru, ar, ms (ar/ms use multilingual model).",
    lifespan=lifespan,
)


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")
    lang: str = Field(..., description=f"Language code. Supported: {SUPPORTED_LANGS}")


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class NERResponse(BaseModel):
    text: str
    lang: str
    entities: list[Entity]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/languages")
def languages():
    return {"supported": SUPPORTED_LANGS, "models": MODELS}


@app.post("/ner", response_model=NERResponse)
def ner(req: NERRequest):
    """Run NER on the given text for the given language."""
    try:
        nlp = get_nlp(req.lang)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    doc = nlp(req.text)
    entities = [
        Entity(text=e.text, label=e.label_, start=e.start_char, end=e.end_char)
        for e in doc.ents
    ]
    return NERResponse(text=req.text, lang=req.lang, entities=entities)
