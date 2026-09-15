"""Tests du nettoyage texte (src/data/text_cleaning.py)."""

import numpy as np

from src.data.text_cleaning import build_text_combined, clean_html, preprocess_text


class TestCleanHtml:
    def test_removes_tags(self):
        assert clean_html("<p>Console <b>Nintendo</b></p>") == "Console Nintendo"

    def test_decodes_entities(self):
        # les entites sont decodees (pas blanchies) : &amp; -> &, &#233; -> é
        assert clean_html("Jeu &amp; console &#233; discount") == "Jeu & console é discount"

    def test_entity_encoded_accent_survives(self):
        # "int&eacute;gr&eacute;" ne doit plus etre tronque en "int gr"
        assert clean_html("int&eacute;gr&eacute;") == "intégré"

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

    def test_accents_normalized_not_stripped(self):
        # v2 : les accents deviennent leur base ASCII, le mot reste entier
        # (avant, "magnetisme" etait tronque en "magn tisme")
        out = preprocess_text("magnétisme protégé", stem=False)
        assert out == "magnetisme protege"

    def test_accented_stopwords_filtered(self):
        # « même »/« était » sont des stopwords FR accentues : ils doivent etre
        # retires malgre la de-accentuation (set de stopwords de-accentue aussi).
        # « console » n'est pas un stopword et survit.
        assert preprocess_text("même était console", stem=False) == "console"

    def test_entity_accent_word_survives_full_pipeline(self):
        # entite HTML -> accent -> de-accentue, le mot reste entier
        out = preprocess_text(clean_html("int&eacute;gration"), stem=False)
        assert out == "integration"

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
