import json
import time
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer

from app.analyzers.base import BaseAnalyzer
from app.analyzers.vision_onnx import OfflineVisionAnalyzer
from app.schemas import AnalysisResult, NsfwSubcategory, Verdict


class HierarchicalVisionAnalyzer(BaseAnalyzer):
    """
    True Two-Stage Offline Image Analyzer:
    Stage 1 (Gatekeeper): Determines SAFE vs. NSFW using Rekib23r ViT + High-Confidence Clothed Kink Gate.
    Stage 2 (Sub-Categorizer): Only runs if Stage 1 is NSFW/UNCERTAIN to identify specific subcategories.
    """
    NSFW_PROMPTS: Dict[NsfwSubcategory, str] = {
        NsfwSubcategory.BONDAGE: "explicit adult bondage fetish photo of a person tied up in shibari ropes, hogtie, chains, or ball gag restraints",
        NsfwSubcategory.BDSM: "explicit adult bdsm fetish photo of a dominatrix, sadomasochism, whipping, dungeon, or collar and leash",
        NsfwSubcategory.FETISH_KINK: "explicit adult fetish photo of latex fetish wear, leather kink costume, or shiny pvc bodysuit",
        NsfwSubcategory.HARDCORE_EXPLICIT: "explicit pornographic photo of hardcore sexual intercourse, sex act, or exposed genitalia",
        NsfwSubcategory.NUDITY_EROTICA: "erotic nude photo of a naked undressed person, topless adult model, or explicit lingerie",
        NsfwSubcategory.HENTAI_ANIME: "explicit hentai anime drawing, nsfw ecchi illustration, or adult cartoon sex",
    }

    # Multiple diverse safe anchors prevent 6-vs-1 softmax dilution on normal photos
    SAFE_PROMPTS: List[str] = [
        "a safe normal everyday photo of a person, portrait, face, selfie, or people in regular clothes",
        "a safe normal photo of nature, landscape, trees, sky, animals, pets, or outdoors",
        "a safe normal photo of an indoor room, furniture, food, car, building, street, or everyday object",
        "a safe normal graphic, solid color background, screenshot, diagram, logo, text, or illustration",
        "a safe normal photo of sports, athletics, gym workout, swimming, or casual fashion",
    ]

    def __init__(
        self,
        clip_dir: str = "models/clip_onnx",
        nsfw_threshold: float = 0.70
    ):
        self._name = "hierarchical_vision_onnx"
        self.nsfw_threshold = nsfw_threshold
        self.base_vision = OfflineVisionAnalyzer()
        self.clip_enabled = False

        clip_path = Path(clip_dir)
        onnx_file = clip_path / "onnx" / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = clip_path / "onnx" / "model.onnx"

        self.image_mean, self.image_std = self._load_preprocessor_config(clip_path)

        if onnx_file.exists():
            try:
                self._patch_config_file(clip_path)
                self.tokenizer = AutoTokenizer.from_pretrained(str(clip_path), local_files_only=True)
                self.session = ort.InferenceSession(
                    str(onnx_file),
                    providers=["CPUExecutionProvider"]
                )
                self.nsfw_categories = list(self.NSFW_PROMPTS.keys())
                all_prompt_texts = list(self.NSFW_PROMPTS.values()) + self.SAFE_PROMPTS

                text_inputs = self.tokenizer(
                    all_prompt_texts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="np"
                )
                self.input_ids = text_inputs["input_ids"].astype(np.int64)
                self.attention_mask = text_inputs["attention_mask"].astype(np.int64)
                self.clip_enabled = True
            except Exception as exc:
                print(f"[Warning] CLIP initialization skipped ({exc}). Using Stage 1 ViT only.")
                self.clip_enabled = False

    @property
    def name(self) -> str:
        return self._name

    def _load_preprocessor_config(self, clip_path: Path) -> tuple[np.ndarray, np.ndarray]:
        default_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        default_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

        config_file = clip_path / "preprocessor_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                mean = np.array(cfg.get("image_mean", default_mean), dtype=np.float32)
                std = np.array(cfg.get("image_std", default_std), dtype=np.float32)
                return mean, std
            except Exception:
                pass
        return default_mean, default_std

    def _patch_config_file(self, clip_path: Path) -> None:
        config_file = clip_path / "config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "model_type" not in cfg:
                cfg["model_type"] = "clip"
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)

    def _preprocess_clip_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.image_mean) / self.image_std
        return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)

    def analyze(self, payload: Union[str, Path, Image.Image]) -> AnalysisResult:
        start_time = time.perf_counter()
        image = Image.open(payload).convert("RGB") if isinstance(payload, (str, Path)) else payload.convert("RGB")

        # --- STAGE 1A: Fast Binary Nudity ViT (Rekib23r) ---
        stage1_res = self.base_vision.analyze(image)
        vit_nsfw_prob = stage1_res.nsfw_probability

        if not self.clip_enabled:
            primary = NsfwSubcategory.NUDITY_EROTICA if stage1_res.verdict == Verdict.NSFW else NsfwSubcategory.NONE
            stage1_res.primary_category = primary
            stage1_res.subcategories = [primary] if primary != NsfwSubcategory.NONE else []
            return stage1_res

        # --- STAGE 1B & STAGE 2: CLIP Evaluation ---
        pixel_values = self._preprocess_clip_image(image)
        feed_dict = {"input_ids": self.input_ids, "pixel_values": pixel_values}
        input_names = {inp.name for inp in self.session.get_inputs()}
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = self.attention_mask

        outputs = self.session.run(None, feed_dict)
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        num_nsfw = len(self.nsfw_categories)
        raw_nsfw_probs = probs[:num_nsfw]
        safe_total_prob = float(np.sum(probs[num_nsfw:]))

        raw_cat_scores = {
            self.nsfw_categories[i].value: round(float(raw_nsfw_probs[i]), 4)
            for i in range(num_nsfw)
        }

        # Clothed Kink Override: Only allow CLIP to override a low Rekib23r score if
        # Bondage + BDSM + Fetish probability is >= 0.75 AND clearly beats all Safe anchors
        clothed_kink_prob = (
            raw_cat_scores[NsfwSubcategory.BONDAGE.value]
            + raw_cat_scores[NsfwSubcategory.BDSM.value]
            + raw_cat_scores[NsfwSubcategory.FETISH_KINK.value]
        )

        if vit_nsfw_prob >= self.nsfw_threshold:
            final_nsfw_prob = vit_nsfw_prob
            verdict = Verdict.NSFW
        elif clothed_kink_prob >= 0.75 and clothed_kink_prob > safe_total_prob:
            final_nsfw_prob = round(clothed_kink_prob, 4)
            verdict = Verdict.NSFW
        elif vit_nsfw_prob <= 0.30 and clothed_kink_prob < 0.50:
            final_nsfw_prob = round(vit_nsfw_prob, 4)
            verdict = Verdict.SAFE
        else:
            final_nsfw_prob = round(max(vit_nsfw_prob, clothed_kink_prob * 0.8), 4)
            verdict = Verdict.UNCERTAIN

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # If SAFE, return immediately without assigning NSFW subcategories
        if verdict == Verdict.SAFE:
            return AnalysisResult(
                analyzer_name=self.name,
                verdict=Verdict.SAFE,
                nsfw_probability=final_nsfw_prob,
                primary_category=NsfwSubcategory.NONE,
                subcategories=[],
                category_scores={},
                latency_ms=latency_ms,
                metadata={
                    "stage1_vit_nudity_prob": vit_nsfw_prob,
                    "stage1_clothed_kink_prob": round(clothed_kink_prob, 4),
                    "safe_anchor_prob": round(safe_total_prob, 4)
                }
            )

        # --- STAGE 2: Normalize Sub-Category Scores Among NSFW Classes ---
        nsfw_sum = float(np.sum(raw_nsfw_probs)) or 1e-6
        normalized_cat_scores = {
            self.nsfw_categories[i].value: round(float(raw_nsfw_probs[i] / nsfw_sum), 4)
            for i in range(num_nsfw)
        }

        sorted_subcats = sorted(normalized_cat_scores.items(), key=lambda item: item[1], reverse=True)
        top_cat_name = sorted_subcats[0][0]
        primary_cat = NsfwSubcategory(top_cat_name)
        detected_subcats = [
            NsfwSubcategory(k) for k, v in sorted_subcats if v >= 0.20
        ]

        return AnalysisResult(
            analyzer_name=self.name,
            verdict=verdict,
            nsfw_probability=final_nsfw_prob,
            primary_category=primary_cat,
            subcategories=detected_subcats,
            category_scores=normalized_cat_scores,
            latency_ms=latency_ms,
            metadata={
                "stage1_vit_nudity_prob": vit_nsfw_prob,
                "stage1_clothed_kink_prob": round(clothed_kink_prob, 4),
                "safe_anchor_prob": round(safe_total_prob, 4)
            }
        )