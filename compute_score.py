from PIL import Image
import xgboost as xgb
import numpy as np
from typing import List, Optional
import imagehash
import io
import base64
from pydantic import BaseModel
import torch
import clip

class ImageHashes(BaseModel):
    average_hash: str = ""
    perceptual_hash: str = ""
    difference_hash: str = ""
    color_hash: str = ""

class ImageResponseBody(BaseModel):
    image_b64: Optional[str] = None
    is_nsfw: Optional[bool] = None
    clip_embeddings: Optional[List[float]] = None
    image_hashes: Optional[ImageHashes] = None

def response_postprocess(image: Image.Image) -> ImageResponseBody:
    image_hashes = image_hash_feature_extraction(image)
    image_b64 = pil_to_base64(image)
    clip_embeddings_of_image = get_clip_embeddings(images=[image])[0]
    return ImageResponseBody(
        image_b64=image_b64,
        image_hashes=image_hashes,
        clip_embeddings=clip_embeddings_of_image,
    )

def image_hash_feature_extraction(image: Image.Image) -> ImageHashes:
    phash = str(imagehash.phash(image))
    ahash = str(imagehash.average_hash(image))
    dhash = str(imagehash.dhash(image))
    chash = str(imagehash.colorhash(image))

    return ImageHashes(
        perceptual_hash=phash,
        average_hash=ahash,
        difference_hash=dhash,
        color_hash=chash,
    )

def get_clip_embeddings(images: Image) -> [List[float]]:
    clip_device = "cuda"
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=clip_device)
    images = [clip_preprocessor(image) for image in images]
    images_tensor = torch.stack(images).to(clip_device)
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(images_tensor)
    image_embeddings = image_embeddings.cpu().numpy().tolist()
    return image_embeddings

def get_clip_embedding_similarity(clip_embedding1: List[float], clip_embedding2: List[float]):
    image_embedding1 = np.array(clip_embedding1, dtype=float).flatten()
    image_embedding2 = np.array(clip_embedding2, dtype=float).flatten()

    norm1 = np.linalg.norm(image_embedding1)
    norm2 = np.linalg.norm(image_embedding2)

    if norm1 == 0 or norm2 == 0:
        return float(norm1 == norm2)

    dot_product = np.dot(image_embedding1, image_embedding2)
    normalized_dot_product = dot_product / (norm1 * norm2)

    return float(normalized_dot_product)

def get_hash_distances(hashes_1: ImageHashes, hashes_2: ImageHashes) -> List[int]:
    ahash_distance = _hash_distance(hashes_1.average_hash, hashes_2.average_hash)
    phash_distance = _hash_distance(hashes_1.perceptual_hash, hashes_2.perceptual_hash)
    dhash_distance = _hash_distance(hashes_1.difference_hash, hashes_2.difference_hash)
    chash_distance = _hash_distance(hashes_1.color_hash, hashes_2.color_hash, color_hash=True)

    return [phash_distance, ahash_distance, dhash_distance, chash_distance]

def _hash_distance(hash_1: str, hash_2: str, color_hash: bool = False) -> int:
    if color_hash:
        restored_hash1 = imagehash.hex_to_flathash(hash_1, hashsize=3)
        restored_hash2 = imagehash.hex_to_flathash(hash_2, hashsize=3)
    else:
        restored_hash1 = imagehash.hex_to_hash(hash_1)
        restored_hash2 = imagehash.hex_to_hash(hash_2)

    return restored_hash1 - restored_hash2

def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def compute(img1: Image.Image, img2: Image.Image):

    img1_response = response_postprocess(img1)
    img2_response = response_postprocess(img2)

    images_are_same_classifier = xgb.XGBClassifier()
    images_are_same_classifier.load_model("image_similarity_xgb_model.json")

    clip_embedding_similiarity = get_clip_embedding_similarity(
        img1_response.clip_embeddings, img2_response.clip_embeddings
    )   
    print(f"Clip embedding similarity: {clip_embedding_similiarity}")

    hash_distances = get_hash_distances(
        img1_response.image_hashes, img2_response.image_hashes
    )
    print(f"Hash distances: {hash_distances}")

    probability_same_image_xg = images_are_same_classifier.predict_proba(
            [hash_distances]
    )[0][1]

    print(f"Probability same image xg: {probability_same_image_xg}")

    # MODEL has a very low threshold
    score = float(probability_same_image_xg**0.5) * 0.4 + (clip_embedding_similiarity**2) * 0.6
    if score > 0.95:
        score = 1

    print(f"Final score: {score**2}")

    return hash_distances, probability_same_image_xg, score**2



