# Efficiency Notes

## LLM Output Token Compression (March 2026)

To reduce the completion token cost of LLM calls, the structured output fields in the image description and review assessment responses were compressed. Specifically, the 2 top-level keys were shortened, safety assessment field names were abbreviated (e.g. `misidentification_risk_people` → `misid_risk`), and enum values were replaced with short codes (e.g. `INCONSISTENT` → `INCON`). Expander modules (like `safety_form_expander.py`) translate the abbreviated output back to full-length keys and values before the data is returned to the rest of the application.

Limited testing (using two images around 10 times each) showed a reduction of roughly 180 completion tokens per image description call (~30%), though the high natural variance in description length made it difficult to isolate the effect precisely. The review assessment showed no clear improvement. The data was too noisy to draw firm conclusions, but the structural argument holds that shorter fixed-length fields will always emit fewer tokens.
