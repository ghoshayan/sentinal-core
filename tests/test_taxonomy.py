from app.schemas import NsfwSubcategory
from app.utils.taxonomy import classify_text_taxonomy


def test_classify_text_taxonomy_bdsm():
    text = "Explore the world of bdsm and shibari bondage."
    primary, subcats, scores, matches = classify_text_taxonomy(text)
    assert primary in [NsfwSubcategory.BDSM, NsfwSubcategory.BONDAGE]
    assert NsfwSubcategory.BDSM in subcats or NsfwSubcategory.BONDAGE in subcats
    assert len(matches) >= 2


def test_classify_text_taxonomy_clean():
    text = "A scientific overview of cellular biology and plant life."
    primary, subcats, scores, matches = classify_text_taxonomy(text)
    assert primary == NsfwSubcategory.NONE
    assert subcats == []
    assert scores == {}
    assert matches == []
