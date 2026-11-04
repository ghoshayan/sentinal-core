from urllib.parse import urljoin
from bs4 import BeautifulSoup

EXPLICIT_KEYWORDS = {
    "porn", "xxx", "sex video", "free porn", "hd porn", "nude", "naked",
    "camgirl", "live sex", "hentai", "erotic", "nsfw", "18+ only",
    "explicit content", "adult video", "blowjob", "milf", "hardcore"
}

def extract_candidate_images(soup: BeautifulSoup, base_url: str, max_images: int = 3) -> list[str]:
    urls = []
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        urls.append(urljoin(base_url, og_img["content"]))

    for img in soup.find_all("img", src=True):
        src = img["src"].strip()
        if src.startswith("data:") or src.lower().endswith(".svg"):
            continue
        width = img.get("width")
        if width and width.isdigit() and int(width) < 150:
            continue
        full_url = urljoin(base_url, src)
        if full_url not in urls:
            urls.append(full_url)
        if len(urls) >= max_images:
            break
    return urls