# Efficiency Notes

## LLM Output Token Compression (March 2026)

To reduce the completion token cost of LLM calls, the structured output fields in the image description and review assessment responses were compressed. Specifically, the 2 top-level keys were shortened, safety assessment field names were abbreviated (e.g. `misidentification_risk_people` → `misid_risk`), and enum values were replaced with short codes (e.g. `INCONSISTENT` → `INCON`). Expander modules (like `safety_form_expander.py`) translate the abbreviated output back to full-length keys and values before the data is returned to the rest of the application.

Limited testing (using two images around 10 times each) showed a reduction of roughly 180 completion tokens per image description call (~30%), though the high natural variance in description length made it difficult to isolate the effect precisely. The review assessment showed no clear improvement. The data was too noisy to draw firm conclusions, but the structural argument holds that shorter fixed-length fields will always emit fewer tokens.

## TOON Format (March 2026)

We attempted to use the TOON format instead of JSON for encoding the safety assessment form when passing it to the review assessment step. The goal was to cut down on the number of input tokens, which is the goal of the format. With the current set of fields being passed in, we saw a reduction of 33 input tokens, about 0.66% of input tokens for the request. We concluded it was not worth the added complexity at this point. If we end up sending a lot more JSON to the models in the future we should revisit this, or if the LLMs start supporting it or other similar formats natively, in which case we could use it in responses for a larger gain.

## Prompt garbage collection revisions (March 2026)

We revisited all the base prompts to check for redundancies and unnecessary directives. This resulted in reducing the number of input tokens per prompt: full description by ~9%, alt text by 70%, review by ~55%. No changes were made to how generated content was submitted back to the models.

## Merging alt text into full description step

This change resulted in around 17% total token usage reduction, and a 12% input token usage reduction. We were not sure this would result in cost savings, since the alt next is now generated using the same large model as the full description, but our test run cost nearly 30% less per image ($0.01416 versus $0.02006). It's not clear if this update is responsible for that cost change or if it was the result of no retries occuring, which would most likely depend on reliability of Azure rather than changes on our end. However, the largest number of retries previously came from the alt text prompts, so it's possible the mini model used there was less reliable, or that the model somehow struggled with it as a standalone task.

## Shortening outputs

Our full description and alt text outputs were long, so we revised the limits on each down lower than before and put hard cut offs since the model was frequently exceeding the recommended max length. This resulted in outputs about half the length for each field, which should have a significant impact on the number of output tokens used.

## Exclude dependent safety fields in negative cases

In order to save output tokens, we are asking the model to not return dependent fields like "demographics_described" when people=N or "text_characteristics.sensitivity" when text_present=N, since those dependent fields would always be set to the same negative values. The default values for the fields are added back into the response from our API, so there is no external change.