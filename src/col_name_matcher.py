from typing import List, Optional, Iterable
import re
import numpy as np

_CAMEL = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])')


class ColNameMatcher:
    model = None
    threshold = 0.80 
    _emb_cache = {} 

    @classmethod
    def init(cls, model, threshold: float = 0.95) -> None:
        cls.model = model
        cls.threshold = threshold
        cls._emb_cache = {}

    @classmethod
    def _preprocess(cls, name: str) -> str:
        s = _CAMEL.sub(' ', str(name))        
        return s.replace('_', ' ').replace('-', ' ').strip().lower()

    @classmethod
    def precompute(cls, col_names: Iterable[str]) -> None:
        if cls.model is None:
            raise RuntimeError("ColNameMatcher.init must be called before precompute")
        unique = []
        seen = set()
        for c in col_names:
            pp = cls._preprocess(c)
            if not pp or pp in cls._emb_cache or pp in seen:
                continue
            seen.add(pp)
            unique.append(pp)
        if not unique:
            return
        embs = cls.model.encode(
            unique,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for pp, e in zip(unique, embs):
            cls._emb_cache[pp] = e

    @classmethod
    def _embed(cls, name: str) -> np.ndarray:
        pp = cls._preprocess(name)
        if pp not in cls._emb_cache:
            cls._emb_cache[pp] = cls.model.encode(
                [pp],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        return cls._emb_cache[pp]

    @classmethod
    def find_match(cls, target: str, candidates) -> Optional[str]:

        candidates = list(candidates)
        if not candidates:
            return None

        target_pp = cls._preprocess(target)

        for c in candidates:
            if cls._preprocess(c) == target_pp:
                return c

        if cls.model is None:
            return None

        target_emb = cls._embed(target)
        cand_embs = np.stack([cls._embed(c) for c in candidates])
        sims = cand_embs @ target_emb
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) >= cls.threshold:
            return candidates[best_idx]
        return None
