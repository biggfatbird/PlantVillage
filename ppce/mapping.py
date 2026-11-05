"""Class alignment between PlantVillage color labels and PlantDoc folder names.

Only pairs with a direct disease or healthy-leaf correspondence are included.
Pairs that would require guessing a different disease are listed in EXCLUDED
and are not used for training or scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassPair:
    plantvillage: str
    plantdoc: str
    kind: str
    note: str


ALIGNED: tuple[ClassPair, ...] = (
    ClassPair("Apple___Apple_scab", "Apple Scab Leaf", "disease", "Same disease name."),
    ClassPair("Apple___Cedar_apple_rust", "Apple rust leaf", "disease", "Same disease name."),
    ClassPair("Apple___healthy", "Apple leaf", "healthy", "PlantDoc healthy leaves are named '<crop> leaf'."),
    ClassPair("Blueberry___healthy", "Blueberry leaf", "healthy", "Healthy-leaf synonym."),
    ClassPair(
        "Cherry_(including_sour)___healthy",
        "Cherry leaf",
        "healthy",
        "PlantVillage cherry class includes sour cherry; PlantDoc does not say so.",
    ),
    ClassPair(
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn Gray leaf spot",
        "disease",
        "Same disease name.",
    ),
    ClassPair("Corn_(maize)___Common_rust_", "Corn rust leaf", "disease", "Same disease name."),
    ClassPair(
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn leaf blight",
        "disease",
        "Assumption: PlantDoc's generic 'Corn leaf blight' is treated as northern leaf blight. "
        "This is a label risk, not a verified one-to-one diagnosis.",
    ),
    ClassPair("Grape___Black_rot", "grape leaf black rot", "disease", "Same disease name."),
    ClassPair("Grape___healthy", "grape leaf", "healthy", "Healthy-leaf synonym. PlantDoc keeps 'grape' lowercase."),
    ClassPair("Peach___healthy", "Peach leaf", "healthy", "Healthy-leaf synonym."),
    ClassPair("Pepper,_bell___healthy", "Bell_pepper leaf", "healthy", "Healthy-leaf synonym."),
    ClassPair("Potato___Early_blight", "Potato leaf early blight", "disease", "Same disease name."),
    ClassPair("Potato___Late_blight", "Potato leaf late blight", "disease", "Same disease name."),
    ClassPair("Raspberry___healthy", "Raspberry leaf", "healthy", "Healthy-leaf synonym."),
    ClassPair(
        "Soybean___healthy",
        "Soyabean leaf",
        "healthy",
        "Healthy-leaf synonym. PlantDoc spells the crop 'Soyabean'.",
    ),
    ClassPair("Squash___Powdery_mildew", "Squash Powdery mildew leaf", "disease", "Same disease name."),
    ClassPair("Strawberry___healthy", "Strawberry leaf", "healthy", "Healthy-leaf synonym."),
    ClassPair("Tomato___Bacterial_spot", "Tomato leaf bacterial spot", "disease", "Same disease name."),
    ClassPair("Tomato___Early_blight", "Tomato Early blight leaf", "disease", "Same disease name."),
    ClassPair("Tomato___Late_blight", "Tomato leaf late blight", "disease", "Same disease name."),
    ClassPair("Tomato___Leaf_Mold", "Tomato mold leaf", "disease", "Same disease name."),
    ClassPair("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot", "disease", "Same disease name."),
    ClassPair(
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato two spotted spider mites leaf",
        "disease",
        "Same condition name.",
    ),
    ClassPair(
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato leaf yellow virus",
        "disease",
        "PlantDoc shortens yellow leaf curl virus to 'yellow virus'.",
    ),
    ClassPair("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus", "disease", "Same disease name."),
    ClassPair("Tomato___healthy", "Tomato leaf", "healthy", "Healthy-leaf synonym."),
)


EXCLUDED: tuple[tuple[str, str], ...] = (
    ("Apple___Black_rot", "No PlantDoc class with this disease name."),
    ("Cherry_(including_sour)___Powdery_mildew", "PlantDoc has no cherry powdery-mildew class."),
    ("Corn_(maize)___healthy", "PlantDoc has no healthy corn class."),
    ("Grape___Esca_(Black_Measles)", "No equally specific PlantDoc class."),
    ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "No equally specific PlantDoc class."),
    ("Orange___Haunglongbing_(Citrus_greening)", "PlantDoc has no citrus class."),
    ("Peach___Bacterial_spot", "PlantDoc peach images are healthy leaves only."),
    (
        "Pepper,_bell___Bacterial_spot",
        "PlantDoc 'Bell_pepper leaf spot' does not state bacterial spot, so the pair is not used.",
    ),
    ("Potato___healthy", "PlantDoc has no healthy potato class."),
    ("Strawberry___Leaf_scorch", "PlantDoc has no strawberry leaf-scorch class."),
    ("Tomato___Target_Spot", "PlantDoc has no tomato target-spot class."),
    ("Bell_pepper leaf spot", "Left unmapped. See Pepper,_bell___Bacterial_spot."),
)


PV_TO_INDEX: dict[str, int] = {pair.plantvillage: i for i, pair in enumerate(ALIGNED)}
INDEX_TO_PV: dict[int, str] = {i: name for name, i in PV_TO_INDEX.items()}
PD_TO_PV: dict[str, str] = {pair.plantdoc: pair.plantvillage for pair in ALIGNED}
CANONICAL_LABELS: tuple[str, ...] = tuple(pair.plantvillage for pair in ALIGNED)


def assert_alignment_is_bijective() -> None:
    pv = [pair.plantvillage for pair in ALIGNED]
    pd = [pair.plantdoc for pair in ALIGNED]
    if len(pv) != len(set(pv)):
        raise ValueError("Duplicate PlantVillage label in ALIGNED")
    if len(pd) != len(set(pd)):
        raise ValueError("Duplicate PlantDoc label in ALIGNED")
    if set(PV_TO_INDEX) != set(pv):
        raise ValueError("PV_TO_INDEX does not match ALIGNED")
