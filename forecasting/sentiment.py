import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from forecasting.data import FNSPID_DIR, copy_to_parquet
from forecasting.news import CACHE_NEWS_DIR, load_or_build_news

FINBERT_MODEL = "ProsusAI/finbert"
# Cached scores per news text column: headlines, and the LSA summaries of the article
# bodies (which only exist for some articles before 2021, and all of them after).
SENTIMENT_CACHES = {
    "title": FNSPID_DIR / "news_sentiment.parquet",
    "lsa_summary": FNSPID_DIR / "news_summary_sentiment.parquet",
}
CHUNK_SIZE = 100_000
# Fastest for summaries (~130 tokens), and close to the fastest for headlines.
BATCH_SIZE = 128
# FinBERT's limit. Headlines are far shorter; 99% of summaries are under ~1,200
# characters (~300 tokens), the few longer ones are truncated.
MAX_TOKENS = 512


def sentiment_chunks_dir(column: str) -> Path:
    """
    Scored chunks of `column` texts, so an interrupted run resumes where it stopped.
    """

    cache = SENTIMENT_CACHES[column]
    return cache.with_name(f"{cache.stem}_chunks")


def load_finbert(
    device: str | None = None,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Load the FinBERT tokenizer and model, in fp16 on the GPU.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    return tokenizer, model.to(device).eval()


@torch.inference_mode()
def score_texts(
    texts: list[str], tokenizer, model, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """
    FinBERT class probabilities for each text, in the input order, plus
    `sentiment = p_positive - p_negative` (in [-1, 1]).

    Texts are batched by length so each batch carries little padding.
    """

    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    order = np.argsort([len(text) for text in texts], kind="stable")
    probs = np.empty((len(texts), len(labels)), dtype=np.float32)

    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer(
            [texts[i] for i in idx],
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(model.device)
        # Softmax in float32, fp16 logits lose precision near 0 and 1.
        probs[idx] = model(**batch).logits.float().softmax(dim=-1).cpu().numpy()

    scores = pd.DataFrame(probs, columns=[f"p_{label}" for label in labels])
    scores.insert(0, "text", texts)
    scores["sentiment"] = scores["p_positive"] - scores["p_negative"]
    return scores


def score_news_texts(column: str, chunk_size: int = CHUNK_SIZE) -> None:
    """
    Score every distinct, non-null `column` text of the news not already in its chunks
    directory, writing one parquet file per `chunk_size` texts. Scores depend only on
    the text, so a headline or summary repeated across symbols or days is scored once.
    """

    if not CACHE_NEWS_DIR.exists():
        load_or_build_news(columns=["symbol"])
    chunks_dir = sentiment_chunks_dir(column)
    chunks_dir.mkdir(exist_ok=True)
    chunks = list(chunks_dir.glob("chunk_*.parquet"))
    # New chunks are numbered after the highest existing one, not by the number of
    # chunks, so a gap in the numbering can't make a new chunk overwrite an old one.
    next_chunk = max((int(path.stem.split("_")[1]) for path in chunks), default=-1) + 1

    con = duckdb.connect()
    done = (
        f"SELECT {column} FROM '{chunks_dir}/chunk_*.parquet'"
        if chunks
        else f"SELECT NULL::VARCHAR AS {column} WHERE false"
    )
    (n_done,) = con.execute(f"SELECT count(*) FROM ({done})").fetchall()[0]
    remaining = con.execute(
        f"""
        SELECT DISTINCT {column} FROM '{CACHE_NEWS_DIR}'
        ANTI JOIN ({done}) USING ({column})
        WHERE {column} IS NOT NULL
        ORDER BY {column}
        """
    )
    texts = [text for (text,) in remaining.fetchall()]
    print(
        f"{len(texts):,} {column} texts to score, {n_done:,} already scored",
        flush=True,
    )
    if not texts:
        return

    tokenizer, model = load_finbert()
    started = time.perf_counter()
    for n, start in enumerate(range(0, len(texts), chunk_size), start=next_chunk):
        scores = score_texts(
            texts[start : start + chunk_size], tokenizer, model
        ).rename(columns={"text": column})
        path = chunks_dir / f"chunk_{n:05d}.parquet"
        tmp_path = path.with_suffix(".tmp")
        scores.to_parquet(tmp_path, index=False)
        tmp_path.rename(path)

        scored = start + len(scores)
        rate = scored / (time.perf_counter() - started)
        print(
            f"{scored:,}/{len(texts):,} texts, {rate:,.0f}/s, "
            f"{(len(texts) - scored) / rate / 60:.1f} min left",
            flush=True,
        )


def load_or_build_news_sentiment(
    column: str = "title",
    force_build=False,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    FinBERT scores per distinct news `column` text (join to the news on `column`),
    cached in SENTIMENT_CACHES[column].
    """

    cache = SENTIMENT_CACHES[column]
    chunks_dir = sentiment_chunks_dir(column)
    if not cache.exists() or force_build:
        if force_build:
            shutil.rmtree(chunks_dir, ignore_errors=True)
        score_news_texts(column)
        copy_to_parquet(
            duckdb.connect(),
            f"SELECT * FROM '{chunks_dir}/chunk_*.parquet' ORDER BY {column}",
            cache,
        )
        shutil.rmtree(chunks_dir)

    return pd.read_parquet(cache, columns=columns)


if __name__ == "__main__":
    for column in SENTIMENT_CACHES:
        load_or_build_news_sentiment(column, columns=[column])
