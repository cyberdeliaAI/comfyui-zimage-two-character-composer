# Z-Image Turbo Two Character Composer

<p align="center">
  <img src="assets/icon.png" alt="Two Character Composer icon" width="200">
</p>

[![Tests](https://github.com/cyberdeliaAI/comfyui-zimage-two-character-composer/actions/workflows/tests.yml/badge.svg)](https://github.com/cyberdeliaAI/comfyui-zimage-two-character-composer/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

One ComfyUI node for **two Z-Image Turbo characters, each with its own prompt and character LoRA**.
Choose left/right, top/bottom, freely positioned boxes or connected masks.
This is a standalone extension: the Krea2 Composer is not required.

**Public beta, 0.2.1.** CPU tensor tests and the real ComfyUI hook/sampler loop
are checked. Full Z-Image GPU generation and the interface in a running ComfyUI
installation still need validation with real checkpoints and character LoRAs.

## Install

Run inside `ComfyUI/custom_nodes`:

```sh
git clone https://github.com/cyberdeliaAI/comfyui-zimage-two-character-composer.git
```

Or download the installation ZIP from [Releases](https://github.com/cyberdeliaAI/comfyui-zimage-two-character-composer/releases).
Extract it so `__init__.py` is directly inside
`custom_nodes/comfyui-zimage-two-character-composer`, without a second nested folder.
Restart ComfyUI, refresh the browser and search for **Z-Image Turbo Two Character Composer**.
No extra Python packages are required beyond a recent, working ComfyUI installation.

Use the standard native diffusion-model loader. Specialized loaders such as
Nunchaku and GGUF have not been validated with these weight hooks.

Model files, following the [official Z-Image Turbo guide](https://docs.comfy.org/tutorials/image/z-image/z-image-turbo):

| File | Folder |
|---|---|
| `z_image_turbo_bf16.safetensors` | `ComfyUI/models/diffusion_models` |
| `qwen_3_4b.safetensors` | `ComfyUI/models/text_encoders` |
| `ae.safetensors` | `ComfyUI/models/vae` |
| Your **Z-Image Turbo** character LoRAs | `ComfyUI/models/loras` |

Load the text encoder with **CLIPLoader → lumina2**. Use **EmptySD3LatentImage**
for the 16-channel latent. Krea2, SDXL and pixel-space Z-Image models are rejected.
The extension does not download models or files at runtime.

## First image: left 60%, right 40%

Import [two_characters.json](example_workflows/two_characters.json) into ComfyUI.
Choose your own model files, then:

1. Enter the shared scene, lighting and style in **Base prompt**.
2. Choose Character 1's LoRA and recommended strength. Put its trigger in
   **Character 1 trigger** and its appearance, outfit and pose in its prompt.
3. Repeat for Character 2 with its own LoRA and trigger.
4. Set **Direction → Left / Right**, **Character 1 percent → 60**,
   **Separation → Strong**, and **Region edge softness → 0**.
5. Generate using the sampler settings that work for your Z-Image Turbo model.
   The example starts at 9 steps, CFG 1, Euler and simple.

Character 1 gets the left 60% and Character 2 the right 40%. For top/bottom,
Character 1 gets the top share. Percentages describe regions, not body sizes.
The boundary snaps to the model's 16px patch grid.

## Connections

| Composer input | Source |
|---|---|
| `model` | Z-Image model, optionally after shared style LoRAs |
| `clip` | Z-Image Qwen3 text encoder |
| `latent` | EmptySD3LatentImage or a compatible existing latent |
| `character_1_mask`, `character_2_mask` | Optional MASK inputs |
| `width`, `height` | Optional preview dimensions |

| Composer output | Destination |
|---|---|
| `model` | KSampler.model |
| `positive` | KSampler.positive |
| `negative` | KSampler.negative |
| `latent` | KSampler.latent_image |
| `region_masks` | Optional Mask To Image → Preview Image |
| `prompts` | Optional Preview Any, to inspect processed prompts |

Use the Composer's **model and positive together**. Character LoRAs belong in
its menus, rather than a global LoRA loader before it. A globally loaded
character LoRA would also affect the other character and background.

## Normal workflow or Composer: one switch

Import [toggle_composer.json](example_workflows/toggle_composer.json).
The **Composer enabled · OFF / ON** Boolean drives four native lazy switches:

- **true / ON:** Composer model, positive, negative and latent.
- **false / OFF:** normal model, normal prompt encoding, empty negative and shared latent.

All routes change together. Do not change the individual routing switches or
bypass only part of the Composer. The shared scene text is also the normal prompt;
edit it for the scene you want. Character settings stay saved while Composer is off.

## Free boxes, masks and overlap

Choose **Free boxes** and drag the rectangles in the preview. Drag a corner to
resize a box, or enter coordinates as percentages:

```text
5, 10, 40, 55; 35, 5, 60, 85
```

Each group means `x, y, width, height`, measured from the upper-left corner.
Keep each box inside the image. Clear the field to restore the default boxes.
The split slider is unused in this mode.

Character 1 has mask priority in an overlap. This controls the contribution of
the regional prediction, not guaranteed foreground depth. Describe who is in
front or behind in the prompts. Outside both character areas, a separate
base-prompt prediction supplies the background without character hooks.

An attached MASK replaces that character's box or split region. White indicates
the active area. The first mask in a batch is used, resized to the latent.
Strong thresholds mask greys at 0.5; Balanced preserves grey weighting.
Empty, faint or fully covered regions produce a clear error. The rectangle
preview does not show the exact silhouette of an input mask; inspect `region_masks`.

## Separation and edge softness

- **Strong:** hard, separate masks. Start here for the strongest identity separation.
- **Balanced:** blends the regional predictions over the number of image pixels
  selected in **Region edge softness**. `0` keeps a hard boundary; try `32` cautiously.

**Z-Image softness affects both prompt and LoRA predictions.** This differs from
Krea2's LoRA-only edge softness. Larger values allow more appearance blending
near boundaries. This node does not implement Krea2's experimental Soft schedule,
Base strength or image-attention isolation.

## Preview ratio and Swap characters

**Auto** tries to read visible width/height values and recognizable resolution
labels. Connect both optional dimension inputs when useful, or choose a fixed
ratio such as `4:5`, `16:9` or `1:1`. Calculated-only sizes may be unavailable
before execution. Generation always uses the actual input LATENT; a preview
preset does not resize it.

**Swap characters** swaps the prompts, triggers, LoRAs and strengths. Free-box
coordinates stay fixed, so the figures exchange locations and box sizes.
In a split, the percentage is also mirrored so each figure retains its share.
Connected input links, including masks, are not swapped.

## Common Prompts and inline LoRAs

Define shared text once, then reuse it:

```text
Base prompt: A garden, {outfit: navy uniform, red scarf}
Character 1: a woman with blonde hair, {outfit}
Character 2: a man with black hair, {outfit}
```

`{name: text}` defines and inserts the text. `{name}` recalls it. Undefined
references stay as text. Conflicting definitions are rejected.

Character trigger/prompt fields also accept `<lora:name:0.8>` tags. The name must
match an installed LoRA unambiguously. Tags are removed before encoding.
If the same file is selected in the menu, the menu strength wins. Tags in the
base or negative prompt are rejected; use a global loader for shared style LoRAs.

## Optional NegPiP negatives

At CFG 1, a conventional negative prompt is ineffective. For the Composer's
negative field and per-character negative weights, install
[NegPiP Prompt](https://github.com/cyberdeliaAI/comfyui-negpip-zimage) separately.
The Composer calls its Z-Image patch; its code is not bundled here.

Use `watermark, blurry` in **Negative prompt** for the whole image, or
`a man, (beard:-0.6)` in one character prompt for a regional negative.
Without negatives, this extension has no dependency on NegPiP. Only encodings
that need negatives use the patched text encoder. A separate NegPiP Prompt node
is not needed in front of Composer. The negative output is a matching zeroed
conditioning, with the passive prompt metadata and NegPiP token data removed.

The bridge checks the installed NegPiP API. Real-model NegPiP generation together
with regional hooks still needs GPU validation in this beta.

## Saved image metadata

Use **Cyberdelia Image Metadata 2.3.1+** and its
[Save Image With Metadata node](https://github.com/cyberdeliaAI/comfyui-cyberdelia-metadata).
It records the shared scene, both character prompts, original global negative,
regional LoRAs/strengths and region settings. No wire from `prompts` is required.
The passive payload survives execution-cache reuse and is batch-aware.

The reader also handles the normal/Composer switch. A latent-only connection
or a mask/report preview does not make an unused Composer an active prompt source.

## Performance and limits

Each character's prompt and hook group is evaluated as its own full-latent
prediction. Only its masked output contributes to the combined result.
With two active character LoRAs, expect two model predictions per step; Free
boxes or masks leaving background space can require a third prediction.
Actual time depends on your hardware, resolution, LoRAs and model loader.

Native **MinVram** hook mode avoids caching extra full weight copies, at the cost
of switching hook weights. No speed-up relative to a plain Turbo workflow is claimed.
Image quality, exact placement and perfect identity separation are not guaranteed.
Start with separated subjects and hard masks before attempting overlap or interaction.
Still images and batches are supported; video is not.

## Updating and checks

Run `git pull` in the extension folder, restart ComfyUI and refresh the browser.
Saved 0.1 prototype fields migrate to the new visual order. The first four
output slots remain model, positive, latent and masks; negative/report are appended.

With PyTorch, NumPy and Node.js installed:

```sh
python -m unittest discover -s tests -v
node --test tests/frontend.test.mjs
python tests/check_native_sampler.py /path/ComfyUI/comfy/hooks.py /path/ComfyUI/comfy/samplers.py /path/ComfyUI/node_helpers.py
```

The native integration check uses real ComfyUI hook attachment and sampler
aggregation with two rank-one tensor LoRAs. Parsing, CLIP and diffusion-model
execution use test doubles. It checks 396 split layouts, four Free boxes /
softness layouts and a combined inline-LoRA case. CI pins ComfyUI commit
[`1332fabfcc507aa7c548cd5152071ed494c207e8`](https://github.com/Comfy-Org/ComfyUI/tree/1332fabfcc507aa7c548cd5152071ed494c207e8).
These checks are not a full Z-Image GPU generation benchmark.

## License and credits

MIT, © 2026 Wilfred Helmig. The preview, prompt helpers and astronaut icon are
shared with Cyberdelia's Krea2 Two Character Composer. The regional engine here
uses ComfyUI's native hooks and masks and contains no Krea2 attention patch.
Optional NegPiP support calls the separately installed AGPL-3.0 extension.
Not affiliated with Alibaba, Tongyi Lab or ComfyUI.
