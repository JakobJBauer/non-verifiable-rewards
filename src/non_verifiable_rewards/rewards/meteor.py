"""English METEOR using NLTK's exact/stem/WordNet alignment implementation."""

import warnings
from collections.abc import Sequence
from pathlib import Path

from .base import checked_scores, validate_inputs


def load_wordnet(path=None):
    """Load an extracted WordNet directory or NLTK's installed corpus offline."""
    try:
        if path is None:
            from nltk.corpus import wordnet

            wordnet.ensure_loaded()
            return wordnet
        from nltk.corpus.reader.wordnet import WordNetCorpusReader
        from nltk.data import FileSystemPathPointer
        from nltk.data import path as nltk_paths

        resource = str(Path(path).resolve())
        if resource not in nltk_paths:
            nltk_paths.append(resource)

        class EnglishWordNetReader(WordNetCorpusReader):
            def map_wn(self, version="wordnet"):
                # NLTK otherwise builds an unused multilingual mapping against
                # a second, globally installed WordNet corpus. English synsets
                # need only the explicitly supplied and checksummed directory.
                return None

        with warnings.catch_warnings():
            # This reward is English-only and never uses multilingual WordNet.
            warnings.filterwarnings(
                "ignore",
                message="The multilingual functions are not available with this Wordnet version",
                category=UserWarning,
            )
            return EnglishWordNetReader(FileSystemPathPointer(resource), omw_reader=None)
    except (LookupError, OSError) as exc:
        raise LookupError(
            "METEOR requires English WordNet. Run the experiment download step, "
            "supply wordnet_path pointing to an extracted WordNet corpus, or install "
            "NLTK's wordnet resource before offline scoring."
        ) from exc


class MeteorReward:
    """NLTK METEOR on complete texts, higher-is-better in [0, 1].

    Uses lowercasing, Porter stemming, English WordNet synonyms, and the default
    alpha=0.9, beta=3, gamma=0.5 fragmentation penalty. Tokenization is NLTK's
    word_tokenize(preserve_line=True), requiring no Punkt resource. Each text
    pair is scored independently, with no sentence splitting or truncation.
    This is NLTK's variant, not the Java METEOR release with paraphrase tables.
    Resources load lazily; scoring never downloads or disables synonym matching.
    """

    variant = "nltk_english_wordnet"

    def __init__(self, *, wordnet_path: str | Path | None = None) -> None:
        self.wordnet_path = wordnet_path
        self._wordnet = None
        self._stemmer = None
        self.statistics: dict | None = None

    def score(
        self,
        candidates: Sequence[str],
        *,
        references: Sequence[str] | None = None,
        prompts: Sequence[str] | None = None,
    ) -> list[float]:
        candidates, references, _ = validate_inputs(
            candidates, references, prompts, needs_references=True
        )
        if not candidates:
            return []
        from nltk import __version__ as nltk_version
        from nltk import word_tokenize
        from nltk.stem import PorterStemmer
        from nltk.translate.meteor_score import single_meteor_score

        if self._wordnet is None:
            self._wordnet = load_wordnet(self.wordnet_path)
            self._stemmer = PorterStemmer(mode=PorterStemmer.NLTK_EXTENSIONS)
            self.statistics = {
                "meteor_variant": self.variant,
                "nltk_version": nltk_version,
                "wordnet_version": self._wordnet.get_version(),
                "tokenizer": "nltk.word_tokenize(preserve_line=True)",
                "lowercase": True,
                "stemmer": "PorterStemmer.NLTK_EXTENSIONS",
                "alpha": 0.9,
                "beta": 3.0,
                "gamma": 0.5,
                "scale": 1,
            }
        # Deduplicate tokenization only within a call, with no unbounded cache.
        tokens = {
            text: word_tokenize(text, preserve_line=True)
            for text in dict.fromkeys(candidates + references)
        }
        return checked_scores(
            [
                single_meteor_score(
                    tokens[reference], tokens[candidate], preprocess=str.lower,
                    stemmer=self._stemmer, wordnet=self._wordnet,
                    alpha=0.9, beta=3.0, gamma=0.5,
                )
                for candidate, reference in zip(candidates, references, strict=True)
            ],
            len(candidates),
        )
