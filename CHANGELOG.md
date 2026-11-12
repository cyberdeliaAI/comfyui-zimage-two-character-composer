# Changelog

## 0.2.1 — beta

### Fixed
- Saved workflows that only carry positional values (frontends that do not write
  `widgets_values_named`) restored most fields into the wrong widgets. The frontend script
  used the Krea2 widget order, not this node's. It now follows the node definition, and a
  test compares the two. The 0.2.0 example workflows list their values in the node's own
  order, and older copies of them are still restored correctly.

## 0.2.0 — first public beta

- Standalone native masked conditioning and independent character LoRA hooks.
- Left/right and top/bottom splits, Free boxes and optional mask inputs.
- Live aspect-ratio preview, draggable rectangles and Swap characters.
- Separate trigger fields, Common Prompts and inline character LoRA tags.
- Optional negatives through the separately installed NegPiP Z-Image patch.
- Appended matching negative and prompt-report outputs.
- Passive runtime metadata for Cyberdelia Image Metadata 2.3.1+.
- Core-only normal/Composer toggle example covering all four generation routes.
- CPU tests against the official ComfyUI hook and sampler loop.

Full-model GPU generation remains to be validated with real Z-Image Turbo LoRAs.

## 0.1.0 — local prototype

- Two character prompts and native LoRA hook groups with split masks.
