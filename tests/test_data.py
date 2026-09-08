"""Tests du nettoyage texte (src/data/text_cleaning.py)."""

import numpy as np

from src.data.text_cleaning import build_text_combined, clean_html, preprocess_text


class TestCleanHtml:
    def test_removes_tags(self):
        assert clean_html("<p>Console <b>Nintendo</b></p>") == "Console Nintendo"

    def test_removes_entities(self):
        assert clean_html("Jeu &amp; console &#233; discount") == "Jeu console discount"

    def test_normalizes_spaces(self):
        assert clean_html("  a   b \n c ") == "a b c"

    def test_nan_returns_empty(self):
        assert clean_html(np.nan) == ""
        assert clean_html(None) == ""


class TestPreprocessText:
    def test_lowercase_and_non_alpha(self):
        out = preprocess_text("Console PS5 Edition 2024 !!!", stem=False)
        assert out == "console edition"  # ps5/2024 non alpha apres filtre, stop words retires

    def test_removes_stopwords(self):
        out = preprocess_text("le produit avec une console", stem=False)
        assert "produit" not in out  # stopword marketplace
        assert "console" in out

    def test_short_tokens_removed(self):
        assert preprocess_text("un le de la xy", stem=False) == ""

    def test_stemming_french(self):
        out = preprocess_text("consoles portables", stem=True)
        assert out == "consol portabl"

    def test_empty_and_nan(self):
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
        assert preprocess_text(np.nan) == ""


class TestBuildTextCombined:
    def test_combines_designation_description(self):
        out = build_text_combined("<b>Console</b> Nintendo", "manette sans fil")
        assert "consol" in out and "nintendo" in out and "manet" in out

    def test_none_description(self):
        out = build_text_combined("Console Nintendo", None)
        assert out == "consol nintendo"
