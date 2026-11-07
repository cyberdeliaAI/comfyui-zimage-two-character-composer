"""Call the separately installed NegPiP pack; no NegPiP code is bundled."""
import sys


def patch(model, clip):
    for name, module in list(sys.modules.items()):
        node = getattr(module, 'NegPipPrompt', None) if name.endswith('.src.nodes') else None
        if node is not None and hasattr(node, '_patch_zimage') and hasattr(module, 'merge_prompts'):
            patched_clip = clip.clone()
            node._patch_zimage(model, patched_clip)
            return patched_clip, module.merge_prompts, node._validate_zimage_negpip_conditioning
    raise RuntimeError('Negative prompts require the separately installed comfyui-negpip-zimage pack. Install/update it and restart ComfyUI, or clear negative prompts and weights.')
