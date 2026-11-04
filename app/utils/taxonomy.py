import re
from typing import Dict, List, Tuple, Set
from app.schemas import NsfwSubcategory

# Categorized taxonomy mapping specific NSFW sub-categories to domain/text indicators
NSFW_TAXONOMY: Dict[NsfwSubcategory, Set[str]] = {
    NsfwSubcategory.BDSM: {
        "bdsm", "femdom", "maledom", "dominatrix", "sadomasochism", "masochism",
        "sadism", "mistress", "domination", "submissive", "master slave",
        "dungeon", "whipping", "flogging", "spanking", "humiliation", "cbt",
        "domme", "fetlife", "kink"
    },
    NsfwSubcategory.BONDAGE: {
        "bondage", "shibari", "kinbaku", "rope bondage", "hogtie", "hogtied",
        "ballgag", "ball gag", "ring gag", "restraints", "bondage tape",
        "suspension bondage", "mummification", "straightjacket", "cuffed",
        "chained", "bound and gagged", "predicament bondage", "shackles"
    },
    NsfwSubcategory.FETISH_KINK: {
        "fetish", "latex fetish", "leather fetish", "chastity cage", "chastity belt",
        "foot fetish", "petplay", "ponyplay", "puppy play", "corset training",
        "enema", "medical fetish", "nylon fetish", "pvc fetish", "Crossdressing fetish"
    },
    NsfwSubcategory.HARDCORE_EXPLICIT: {
        "porn", "pornhub", "xvideos", "xnxx", "xhamster", "redtube", "youporn",
        "spankbang", "eporner", "brazzers", "xxx", "sex video", "free porn",
        "hd porn", "hardcore", "blowjob", "cumshot", "gangbang", "anal sex",
        "creampie", "milf", "threesome", "adult video", "nsfw"
    },
    NsfwSubcategory.NUDITY_EROTICA: {
        "nude", "naked", "nudity", "topless", "erotic", "erotica", "playboy",
        "onlyfans", "fansly", "adult model", "boudoir nude", "stripper"
    },
    NsfwSubcategory.HENTAI_ANIME: {
        "hentai", "rule34", "r34", "doujinshi", "nhentai", "hanime",
        "ecchi", "futanari", "tentacle hentai", "anime porn"
    },
    NsfwSubcategory.LIVE_CAM: {
        "chaturbate", "stripchat", "camgirl", "cam4", "myfreecams", "bongacams",
        "livejasmin", "live sex", "sex cam", "adult webcam"
    }
}

def classify_text_taxonomy(text: str) -> Tuple[NsfwSubcategory, List[NsfwSubcategory], Dict[str, float], List[str]]:
    """
    Scans URL or HTML text against all NSFW sub-categories.
    Returns: (primary_category, detected_subcategories, category_scores, matched_terms)
    """
    normalized = text.lower()
    scores: Dict[str, float] = {}
    all_matches: List[str] = []

    for category, keywords in NSFW_TAXONOMY.items():
        matched_in_cat = []
        for kw in keywords:
            # Use word-boundary regex so short words like 'cbt' or 'xxx' don't match substrings
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, normalized):
                matched_in_cat.append(kw)

        if matched_in_cat:
            all_matches.extend(matched_in_cat)
            # Calculate confidence score for this specific sub-category (0.50 base + 0.15 per extra match)
            scores[category.value] = round(min(0.55 + (0.15 * len(matched_in_cat)), 0.99), 4)

    if not scores:
        return NsfwSubcategory.NONE, [], {}, []

    # Sort categories by highest score
    sorted_cats = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_cat = NsfwSubcategory(sorted_cats[0][0])
    detected_list = [NsfwSubcategory(cat_name) for cat_name, score in sorted_cats if score >= 0.55]

    return primary_cat, detected_list, scores, list(set(all_matches))