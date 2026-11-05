# Prior-art notes for the confidence-gated edit

These notes are a technical search, not a legal opinion and not a patentability determination. The search used public web indexes of patents and papers in November 2026. It is not a complete search of every patent office.

## What was compared

The mechanism in `ppce/confidence_edit.py` does four things together:

1. It separates environment, leaf body, an uncertain disease band, and a high-agreement disease core with several fixed color rules.
2. It edits environment and unprotected leaf body, and it copies the core and the uncertain band from the source.
3. It re-runs the same rules on the edited image and checks explicit preservation conditions.
4. It refuses to emit the edit, and it records the failed condition, when preservation of the core cannot be established.

## Overlapping techniques

| Source | What overlaps | What it does not do |
| --- | --- | --- |
| US patent application 18/791,123, "Training server and method of augmenting agricultural image data for enhanced error handling," application date 31 July 2024, publication date shown as 24 April 2025 on the indexed copy (identifier 20250133177). | Detects foliage versus soil and builds an augmented training image by changing soil hue while leaving foliage pixels unaltered in RGB. | The protected region is foliage as a whole. There is no disease core, no uncertain disease band, and no rejection of an edit when disease evidence is uncertain. The stated task is crop or foliage detection. |
| US 12,079,989 B2, "Identifying boundaries of lesions within image data." Indexed at https://patents.google.com/patent/US12079989B2/en | Produces lesion probability and uncertainty and can reject a candidate lesion when its uncertainty is high. | Rejection removes a detection. It is not a gate on whether an image edit may be used for domain adaptation, and it does not freeze an uncertain band inside an edited image. |
| US 10,460,448, contour-uncertainty quantification. Indexed discussion at the retrieved patent summary for that number. | Represents a spatial band of contour uncertainty. | It quantifies delineation uncertainty. It does not accept or reject an environmental edit of a plant image. |
| CN 117197525 B, plant leaf disease detection and identification. https://patents.google.com/patent/CN117197525B/en | Segments a leaf, removes background, augments disease-leaf images, and retrains when segmentation mIoU or classification accuracy is below a stated threshold. | The quality gate retrains a segmenter or classifier. It does not reject one proposed edit for failing disease-pixel preservation. |
| Kim, Jeong, Kim, and Choi, STAug, IEEE Access, 2022, DOI 10.1109/ACCESS.2022.3224141. | Copies a salient plant region onto another background so plant color and shape are kept. The paper explicitly wants augmentation that does not drop blight regions. | The preserved object is a salient-object mask of the plant, not a disease core separated from the leaf. There is no uncertainty band and no per-edit rejection log. |
| Ghiasi et al., Simple Copy-Paste, and Tobin et al., domain randomization. | Change context while retaining foreground pixels. | General augmentation. No plant-disease evidence check. |
| The 2026 counterfactual plant-disease analysis indexed at https://www.scitepress.org/Papers/2026/144727/144727.pdf | Uses lesion masks to edit plant images and measure the effect on a classifier. The edit removes lesions. | The intervention deletes disease evidence. This mechanism tries to keep it, and it rejects the edit when the keep-set is missing or unstable. |
| The first PPCE prototype already in this repository (`ppce/editing.py`). | Copies a single heuristic lesion mask back onto a recolored image. | One mask, no agreement score, no uncertain band, no rejection, and no reason log. On the saved corn example that mask followed the pale midrib. |

## Search limits

Patent-family equivalents, claim charts, and file wrappers were not reviewed. A phrase that did not appear in the snippets above can still exist in a claim. A qualified patent professional has to decide whether any of these disclosures anticipate or make obvious the combination implemented here.
