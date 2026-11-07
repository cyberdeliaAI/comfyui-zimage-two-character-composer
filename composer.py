"""Two regions using ComfyUI's native masked conditioning and model LoRA hooks."""

import math

NO_LORA = "None"
DIRECTIONS = ["Left / Right", "Top / Bottom", "Free boxes"]
SEPARATION = ["Balanced", "Strong"]


def validate_percent(direction, percent):
    if direction not in DIRECTIONS:
        raise ValueError("Choose Left / Right or Top / Bottom.")
    if isinstance(percent, bool):
        raise ValueError("Character 1 percentage must be between 1 and 99.")
    percent = float(percent)
    if not math.isfinite(percent) or not 1 <= percent <= 99:
        raise ValueError("Character 1 percentage must be between 1 and 99.")
    return percent


def latent_size(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    shape = getattr(samples, "shape", ())
    if len(shape) != 4 or shape[1] != 16:
        raise ValueError("Connect a 16-channel still-image LATENT, for example EmptySD3LatentImage.")
    height, width = int(shape[-2]), int(shape[-1])
    if min(height, width) < 8 or height % 2 or width % 2:
        raise ValueError("Z-Image dimensions must be multiples of 16 pixels, at least 64px.")
    return height, width


def region_masks(height, width, direction, percent, separation):
    """Complementary masks on the actual latent grid, not independent rounding."""
    import torch

    percent = validate_percent(direction, percent)
    if separation not in SEPARATION:
        raise ValueError("Choose Balanced or Strong separation.")
    vertical = direction == DIRECTIONS[1]
    size = height if vertical else width
    cut = 2 * max(1, min(size // 2 - 1, round((size // 2) * percent / 100)))
    coordinates = torch.arange(size, dtype=torch.float32) + 0.5
    if separation == "Balanced":
        # Up to 32 image pixels total; keep the transition symmetric even when
        # one region is narrow. Integrated region area still equals `cut`.
        radius = min(2.0, float(cut), float(size - cut))
        line = ((cut + radius - coordinates) / (2 * radius)).clamp(0, 1)
    else:
        line = (coordinates < cut).to(torch.float32)
    first = line[:, None].expand(height, width) if vertical else line[None, :].expand(height, width)
    return first.contiguous(), (1.0 - first).contiguous()


def _runtime():
    import comfy.hooks
    import comfy.lora
    import comfy.utils
    import folder_paths

    required = ("create_hook_lora", "set_hooks_for_conditioning", "set_mask_for_conditioning")
    if (not all(hasattr(comfy.hooks, name) for name in required)
            or not hasattr(getattr(comfy.hooks, "EnumHookMode", None), "MinVram")):
        raise RuntimeError("Update ComfyUI: this Composer needs native conditioning masks and LoRA hooks.")
    return comfy.hooks, comfy.lora, comfy.utils, folder_paths


def _lora_hook(name, strength, key_map, hooks, lora_api, utils, paths):
    strength = float(strength)
    if not math.isfinite(strength) or not -10 <= strength <= 10:
        raise ValueError("LoRA strength must be finite and between -10 and 10.")
    if name == NO_LORA or strength == 0:
        return None
    path = paths.get_full_path_or_raise("loras", name)
    state = utils.load_torch_file(path, safe_load=True)
    patches = lora_api.load_lora(state, key_map)
    if not patches:
        raise ValueError(f"LoRA '{name}' has no matching Z-Image model layers. Choose a Z-Image LoRA and update ComfyUI if needed.")
    # Native hooks consume pre-mapped patches too. Avoid parsing the same file
    # again at sampler registration; text-encoder LoRAs are intentionally off.
    group = hooks.create_hook_lora(lora=state, strength_model=strength, strength_clip=0.0)
    for hook in group.hooks:
        hook.weights = patches
        hook.need_weight_init = False
    return group


from .prompting import (DEFAULT_BOXES, expand_common, region_prompt, relative_sides,
                        extract_lora_tags, has_negative_weights)
from .regions import layout_masks, centroid

METADATA_KEY = 'cyberdelia_composer_metadata'
PREVIEW_RATIOS = ['Auto', '1:1', '4:3', '3:4', '3:2', '2:3', '16:9', '9:16', '21:9', '9:21', '5:4', '4:5']


def character_loras(menu, strength, tags, paths):
    inventory = paths.get_filename_list('loras')
    def resolve(name):
        from pathlib import PurePath
        normalized = name.replace('\\', '/')
        exact = [p for p in inventory if p.replace('\\', '/') == normalized]
        choices = exact or [p for p in inventory if PurePath(p.replace('\\', '/')).stem == PurePath(normalized).stem]
        choices = choices or [p for p in inventory if normalized.lower() in p.replace('\\', '/').lower()]
        if len(choices) != 1:
            raise ValueError(f'LoRA name {name!r} is missing or ambiguous. Choose its exact menu path.')
        return choices[0]
    rows = []
    for name, value in [(menu, strength)] + tags:
        value = float(value)
        if not math.isfinite(value) or not -10 <= value <= 10:
            raise ValueError('LoRA strength must be finite and between -10 and 10.')
        if name == NO_LORA or value == 0:
            continue
        name = resolve(name)
        if not any(n == name for n, _ in rows):
            rows.append((name, value))
    return rows


def zero_conditioning(positive):
    import torch
    out = []
    for tensor, metadata in positive:
        extra = {k: v for k, v in metadata.items() if k != METADATA_KEY and not k.startswith(('negpip_', 'zimage_negpip', 'cyberdelia_negpip_zimage_'))}
        if torch.is_tensor(extra.get('pooled_output')):
            extra['pooled_output'] = torch.zeros_like(extra['pooled_output'])
        out.append([torch.zeros_like(tensor), extra])
    return out


class ZImageTwoCharacterComposer:
    CATEGORY = 'conditioning/zimage'
    FUNCTION = 'compose'
    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'LATENT', 'MASK', 'CONDITIONING', 'STRING')
    RETURN_NAMES = ('model', 'positive', 'latent', 'region_masks', 'negative', 'prompts')
    DESCRIPTION = ('Two regional characters with independent LoRA hook groups. Uses separate full-latent '
                   'model predictions, blended only inside the selected regions. Standard native model loaders required.')

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        loras = [NO_LORA] + [n for n in folder_paths.get_filename_list('loras') if n != NO_LORA]
        text = {'default': '', 'multiline': True}
        required = {'model': ('MODEL',), 'clip': ('CLIP',), 'latent': ('LATENT',),
                    'direction': (DIRECTIONS, {'default': DIRECTIONS[0]}),
                    'character_1_percent': ('INT', {'default': 50, 'min': 1, 'max': 99, 'step': 1, 'display': 'slider'}),
                    'base_prompt': ('STRING', {**text, 'placeholder': 'Shared scene, background, lighting and style'})}
        for n in (1, 2):
            required.update({f'character_{n}_prompt': ('STRING', {**text, 'placeholder': f'Character {n}: appearance, outfit and pose'}),
                             f'character_{n}_lora': (loras, {'default': NO_LORA}),
                             f'character_{n}_strength': ('FLOAT', {'default': 1., 'min': -10., 'max': 10., 'step': .05})})
        required['separation'] = (SEPARATION, {'default': 'Strong', 'tooltip': 'Strong keeps masks separate. Balanced blends regional predictions using edge_softness.'})
        optional = {'negative_prompt': ('STRING', {**text, 'placeholder': 'Optional global negatives (requires NegPiP)'}),
                    'character_1_trigger': ('STRING', {'default': ''}),
                    'character_2_trigger': ('STRING', {'default': ''}),
                    'boxes': ('STRING', {'default': DEFAULT_BOXES}),
                    'edge_softness': ('INT', {'default': 0, 'min': 0, 'max': 256, 'step': 8,
                        'tooltip': 'Balanced only. Image pixels over which regional predictions blend. Affects both prompts and LoRAs; use 0 for strongest separation.'}),
                    'preview_ratio': (PREVIEW_RATIOS, {'default': 'Auto'}),
                    'character_1_mask': ('MASK',), 'character_2_mask': ('MASK',),
                    'width': ('INT', {'forceInput': True}), 'height': ('INT', {'forceInput': True})}
        return {'required': required, 'optional': optional}

    def compose(self, model, clip, latent, direction, character_1_percent,
                base_prompt, character_1_prompt, character_1_lora, character_1_strength,
                character_2_prompt, character_2_lora, character_2_strength, separation,
                negative_prompt='', character_1_trigger='', character_2_trigger='', boxes=DEFAULT_BOXES,
                edge_softness=0, preview_ratio='Auto', character_1_mask=None, character_2_mask=None,
                width=None, height=None):
        import torch
        if direction not in DIRECTIONS or separation not in SEPARATION:
            raise ValueError('Choose a supported direction and separation mode.')
        percent = validate_percent(DIRECTIONS[0] if direction == 'Free boxes' else direction, character_1_percent)
        height, width = latent_size(latent)  # Always derive generation size from LATENT.
        if preview_ratio not in PREVIEW_RATIOS:
            raise ValueError('Choose Auto or one of the fixed preview ratios.')
        if not math.isfinite(float(edge_softness)) or not 0 <= float(edge_softness) <= 256:
            raise ValueError('Region edge softness must be between 0 and 256 pixels.')
        config = getattr(getattr(getattr(model, 'model', None), 'model_config', None), 'unet_config', {})
        if config.get('image_model') != 'lumina2' or config.get('dim') != 3840:
            raise ValueError('Connect a native latent-space Z-Image Turbo MODEL; other model families are incompatible.')
        (base, negative, trigger1, prompt1, trigger2, prompt2), unknown = expand_common([
            base_prompt, negative_prompt, character_1_trigger, character_1_prompt, character_2_trigger, character_2_prompt])
        base, base_tags = extract_lora_tags(base)
        negative, negative_tags = extract_lora_tags(negative)
        if base_tags or negative_tags:
            raise ValueError('LoRA tags belong in a character trigger or prompt. Load global style LoRAs before Composer.')
        characters, tags = [], []
        for trigger, prompt in ((trigger1, prompt1), (trigger2, prompt2)):
            trigger, trigger_tags = extract_lora_tags(trigger)
            prompt, prompt_tags = extract_lora_tags(prompt)
            characters.append(region_prompt(trigger, prompt))
            tags.append(trigger_tags + prompt_tags)
        if not all(characters):
            raise ValueError('Enter a prompt or trigger for both characters.')
        masks, layout = layout_masks(height, width, direction, percent, boxes, character_1_mask, character_2_mask,
                                     int(edge_softness), separation)
        hooks, lora_api, utils, paths = _runtime()
        working_model = model.clone()
        if getattr(working_model, 'is_dynamic', lambda: False)():
            delegate = getattr(working_model, 'get_non_dynamic_delegate', None)
            if not callable(delegate):
                raise RuntimeError('This loader has no native hook delegate. Use the standard diffusion-model loader.')
            working_model = delegate()
        if not all(callable(getattr(working_model, method, None))
                   for method in ('register_all_hook_patches', 'set_hook_mode')):
            raise RuntimeError('This loader does not support native LoRA hooks. Use the standard model loader.')
        working_model.set_hook_mode(hooks.EnumHookMode.MinVram)
        rows = [character_loras(name, strength, additions, paths) for name, strength, additions in (
            (character_1_lora, character_1_strength, tags[0]), (character_2_lora, character_2_strength, tags[1]))]
        key_map = lora_api.model_lora_keys_unet(working_model.model, {})
        groups = []
        for selected in rows:
            parts = [_lora_hook(name, strength, key_map, hooks, lora_api, utils, paths) for name, strength in selected]
            groups.append(hooks.HookGroup.combine_all_hooks(parts) if len(parts) > 1 else parts[0] if parts else None)
        sides = relative_sides(*(centroid(mask) for mask in masks[:2]))
        texts = ['. '.join(p for p in (base.strip().rstrip('.'), 'Two distinct characters in one continuous scene',
                  (f'{sides[i]}: ' if sides else '') + characters[i]) if p) for i in (0, 1)]
        has_background = bool((masks[2] > 0).any())
        if has_background:
            texts.append(base.strip() or 'One continuous background, consistent lighting')
        needs = [bool(negative.strip()) or has_negative_weights(text) for text in texts]
        patched_clip, merge, validate = None, None, None
        if any(needs):
            from .negpip_bridge import patch
            patched_clip, merge, validate = patch(working_model, clip)
        positive, encoded = [], []
        for i, text in enumerate(texts):
            chosen_clip = patched_clip if needs[i] else clip
            encoded_text = merge(text, negative) if needs[i] else text
            encoded.append(encoded_text)
            cond = chosen_clip.encode_from_tokens_scheduled(chosen_clip.tokenize(encoded_text))
            if needs[i]:
                validate(cond)
            if any(extra.get('hooks') is not None for _, extra in cond):
                raise ValueError('The connected CLIP already attaches character hooks. Select character LoRAs in Composer instead.')
            group = groups[i] if i < 2 else None
            cond = hooks.set_hooks_for_conditioning(cond, group, append_hooks=False)
            cond = hooks.set_mask_for_conditioning(cond, masks[i], set_cond_area='default', strength=1.)
            positive.extend(cond)
        metadata = {'version': 1, 'model_family': 'zimage', 'scene_prompt': base,
                    'encoded_scene_prompt': base, 'encoded_region_prompts': encoded,
                    'character_prompts': characters, 'negative_prompt': negative,
                    'loras': [[{'name': n, 'strength': s} for n, s in selected] for selected in rows],
                    'layout': layout, 'separation': separation, 'edge_softness': int(edge_softness),
                    'width': width * 8, 'height': height * 8, 'negpip': any(needs)}
        positive = [[tensor, {**extra, METADATA_KEY: metadata}] for tensor, extra in positive]
        report = '\n'.join([f'Scene: {base}'] + [f'Character {i+1}: {t}' for i, t in enumerate(characters)] +
                           [f'Layout: {layout}', f'Region edge softness: {int(edge_softness)}px ({separation})',
                            'Encoded branches:'] + [f'{i+1}: {t}' for i, t in enumerate(encoded)] +
                           ([f'Undefined common prompts: {", ".join(unknown)}'] if unknown else []))
        previews = masks[:2].repeat_interleave(8, -2).repeat_interleave(8, -1)
        return working_model, positive, latent, previews, zero_conditioning(positive), report
