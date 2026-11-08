import cv2
import numpy as np


def element_ligne(taille: int, angle: float) -> np.ndarray:
    """Créer un élément structurant linéaire selon une orientation."""

    if taille < 3 or taille % 2 == 0:
        raise ValueError("taille doit être impaire et >= 3")

    kernel = np.zeros((taille, taille), dtype=np.uint8)

    c = taille // 2
    rad = np.deg2rad(angle)

    dx = round(c * np.cos(rad))
    dy = round(c * np.sin(rad))

    cv2.line(
        kernel,
        (c - dx, c - dy),
        (c + dx, c + dy),
        1,
        1
    )

    return kernel


def dullrazor(image_rgb: np.ndarray, taille: int = 17, seuil: int = 18):
    """
    Détection des poils candidats par Black-hat multi-orientations
    et reconstruction par inpainting de Telea.

    Retour :
        image_rgb, response, masque, image_nettoyee
    """

    if (
        image_rgb is None
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.uint8
    ):
        raise ValueError("L'image doit être RGB uint8.")

    h, w = image_rgb.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Image vide")

    scale = min(1.0, 768 / w)

    if scale < 1:
        image_rgb = cv2.resize(
            image_rgb,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    response = np.zeros_like(gray)

    for angle in [0, 45, 90, 135]:

        kernel = element_ligne(taille, angle)

        fermeture = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

        difference = cv2.subtract(fermeture, gray)
        response = np.maximum(response, difference)

    _, masque = cv2.threshold(
        response,
        seuil,
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    masque = cv2.dilate(
        masque,
        kernel,
        iterations=1
    )

    image_nettoyee = cv2.inpaint(
        image_rgb,
        masque,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    return image_rgb, response, masque, image_nettoyee