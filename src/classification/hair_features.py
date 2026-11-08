import cv2
import numpy as np


def line_kernel(size, angle):
    """Construire un élément structurant linéaire orienté."""

    k = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    a = np.deg2rad(angle)

    dx = round(c * np.cos(a))
    dy = round(c * np.sin(a))

    cv2.line(
        k, (c-dx, c-dy), (c+dx, c+dy), 1, 1
    )

    return k


def extract_hair_features(image_rgb):
    """
    Extraire des descripteurs mathématiques
    liés aux structures sombres fines de l'image.

    Cette fonction ne nettoie PAS l'image.
    Elle fournit les caractéristiques au classificateur.
    """

    # Même taille d'analyse pour toutes les images
    image = cv2.resize(
        image_rgb, (256, 256),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    responses = []

    # Fermetures morphologiques directionnelles
    for angle in [0, 45, 90, 135]:

        kernel = line_kernel(17, angle)

        closed = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE, kernel
        )

        blackhat = cv2.subtract(closed, gray)
        responses.append(blackhat)

    response = np.maximum.reduce(responses)

    # Masque de structures sombres candidates
    mask = (response > 18).astype(np.uint8)

    # Analyse des composantes connexes
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(mask, 8)
    )

    areas = []
    elongations = []
    linear_areas = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        if area < 3:
            continue

        elongation = max(w, h) / max(1, min(w, h))

        areas.append(area)
        elongations.append(elongation)

        if elongation >= 3:
            linear_areas.append(area)

    # Caractéristiques numériques
    features = [
        float(gray.mean()),
        float(gray.std()),

        float(response.mean()),
        float(response.std()),
        float(np.percentile(response, 90)),
        float(np.percentile(response, 95)),
        float(np.percentile(response, 99)),

        float(mask.mean()),
        float(len(areas)),
        float(np.mean(areas)) if areas else 0.0,
        float(max(areas)) if areas else 0.0,

        float(np.mean(elongations)) if elongations else 0.0,
        float(max(elongations)) if elongations else 0.0,

        float(sum(linear_areas) / mask.size),
    ]

    return np.asarray(features, dtype=np.float32)


FEATURE_NAMES = [
    "gray_mean",
    "gray_std",
    "blackhat_mean",
    "blackhat_std",
    "blackhat_p90",
    "blackhat_p95",
    "blackhat_p99",
    "candidate_pixel_ratio",
    "component_count",
    "mean_component_area",
    "max_component_area",
    "mean_elongation",
    "max_elongation",
    "linear_component_area_ratio"
]