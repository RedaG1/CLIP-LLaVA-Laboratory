# Hallucination example (llava-hf/llava-1.5-7b-hf)

Image: `assets\image.jpg` (contains **no** bicycle)

## Independent questions

- **Q:** Is there a bicycle in this image?
  **A:** No, there is no bicycle in this image. The image features two cats sleeping on a couch.
- **Q:** What color is the bicycle?
  **A:** The bicycle is pink.
- **Q:** Where is the bicycle located?
  **A:** The bicycle is located in the living room.

## Multi-turn conversation

- **Q:** Is there a bicycle in this image?
  **A:** No, there is no bicycle in this image. The image features two cats sleeping on a couch.
- **Q:** What color is the bicycle?
  **A:** The bicycle is blue.
- **Q:** Where is the bicycle located?
  **A:** The bicycle is located on the floor.

## Prompt with an explicit 'absent' option

- **Q:** Look carefully. If there is a bicycle, what color is it? If there is no bicycle, answer 'There is no bicycle'.
  **A:** There is no bicycle.

CLIP probabilities over [cat, dog, bicycle, sofa]: {'cat': np.float32(0.6899), 'dog': np.float32(0.0025), 'bicycle': np.float32(0.0012), 'sofa': np.float32(0.3064)}
