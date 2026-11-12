import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("zcomposer_unit", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
composer = sys.modules[f"{spec.name}.composer"]


class Model:
    def __init__(self, dynamic=False):
        self.model = types.SimpleNamespace(model_config=types.SimpleNamespace(unet_config={"image_model": "lumina2", "dim": 3840}))
        self.dynamic = dynamic
        self.mode = None
    def clone(self):
        return Model(self.dynamic)
    def is_dynamic(self):
        return self.dynamic
    def get_non_dynamic_delegate(self):
        return Model(False)
    def register_all_hook_patches(self, *args):
        pass
    def set_hook_mode(self, mode):
        self.mode = mode


class Clip:
    def __init__(self):
        self.prompts = []
        self.conditionings = []
        self.existing_hooks = None
    def tokenize(self, text):
        self.prompts.append(text)
        return text
    def encode_from_tokens_scheduled(self, tokens):
        extra = {"attention_mask": torch.ones(1, 3)}
        if self.existing_hooks:
            extra["hooks"] = self.existing_hooks
        cond = [[torch.ones(1, 3, 4), extra]]
        self.conditionings.append(cond)
        return cond


def clone_conditioning(cond, values):
    return [[embedding, metadata | values] for embedding, metadata in cond]


class ComposerTests(unittest.TestCase):
    def setUp(self):
        self.loaded = []
        def load(path, safe_load):
            self.assertTrue(safe_load)
            self.loaded.append(path)
            return {"name": path}
        def create(lora, strength_model, strength_clip):
            self.assertEqual(strength_clip, 0)
            return types.SimpleNamespace(hooks=[types.SimpleNamespace(weights=lora, need_weight_init=True, strength=strength_model)])
        self.hooks = types.SimpleNamespace(
            create_hook_lora=create,
            EnumHookMode=types.SimpleNamespace(MinVram="minvram"),
            set_hooks_for_conditioning=lambda cond, group, append_hooks: cond if group is None else clone_conditioning(cond, {"hooks": group}),
            set_mask_for_conditioning=lambda cond, mask, set_cond_area, strength: clone_conditioning(cond, {"mask": mask.unsqueeze(0), "set_area_to_bounds": set_cond_area != "default", "mask_strength": strength}),
        )
        self.lora = types.SimpleNamespace(model_lora_keys_unet=lambda model, mapping: {"layer": "diffusion_model.layer.weight"},
                                          load_lora=lambda state, mapping: {"diffusion_model.layer.weight": state})
        self.utils = types.SimpleNamespace(load_torch_file=load)
        self.paths = types.SimpleNamespace(get_full_path_or_raise=lambda kind, name: name, get_filename_list=lambda kind: ["a.safetensors", "b.safetensors"])
        runtime = patch.object(composer, "_runtime", return_value=(self.hooks, self.lora, self.utils, self.paths))
        self.runtime = runtime.start()
        self.addCleanup(runtime.stop)
        self.args = dict(model=Model(), clip=Clip(), latent={"samples": torch.zeros(2, 16, 12, 16), "noise_mask": object()},
            direction="Left / Right", character_1_percent=60, base_prompt="Two people in a garden",
            character_1_prompt="first_trigger, a red knight", character_1_lora="a.safetensors", character_1_strength=0.75,
            character_2_prompt="second_trigger, a blue wizard", character_2_lora="b.safetensors", character_2_strength=1.25, separation="Strong")

    def compose(self, **overrides):
        return composer.ZImageTwoCharacterComposer().compose(**(self.args | overrides))

    def test_masks_cover_once_for_every_percentage_direction_and_mode(self):
        for direction in composer.DIRECTIONS[:2]:
            for mode in composer.SEPARATION:
                for percent in range(1, 100):
                    first, second = composer.region_masks(16, 24, direction, percent, mode)
                    torch.testing.assert_close(first + second, torch.ones(16, 24))
                    axis = 16 if direction == "Top / Bottom" else 24
                    cut = 2 * max(1, min(axis // 2 - 1, round((axis // 2) * percent / 100)))
                    self.assertAlmostEqual(first.sum().item(), cut * (24 if direction == "Top / Bottom" else 16), places=4)
                    if mode == "Strong":
                        self.assertEqual((first * second).sum().item(), 0)
                    else:
                        self.assertGreater((first * second).sum().item(), 0)

    def test_two_hooks_keep_independent_loras_and_strengths_without_global_patch(self):
        output = self.compose()
        model, positive, latent, masks, negative, report = output
        self.assertIsNot(model, self.args["model"])
        self.assertIsNone(self.args["model"].mode)
        self.assertEqual(model.mode, "minvram")
        self.assertIs(latent, self.args["latent"])
        self.assertEqual(masks.shape, (2, 96, 128))
        self.assertEqual(len(positive), 2)
        groups = [entry[1]["hooks"] for entry in positive]
        self.assertIsNot(groups[0], groups[1])
        self.assertEqual([g.hooks[0].strength for g in groups], [0.75, 1.25])
        self.assertEqual([g.hooks[0].weights["diffusion_model.layer.weight"]["name"] for g in groups], ["a.safetensors", "b.safetensors"])
        self.assertTrue(all(not g.hooks[0].need_weight_init for g in groups))
        self.assertTrue(all(not entry[1]["set_area_to_bounds"] for entry in positive))
        self.assertTrue(all("mask" not in cond[0][1] and "hooks" not in cond[0][1] for cond in self.args["clip"].conditionings))

    def test_prompt_encoding_keeps_each_trigger_in_its_own_branch(self):
        self.compose(direction="Top / Bottom")
        first, second = self.args["clip"].prompts
        self.assertIn("top of the image", first)
        self.assertIn("bottom of the image", second)
        self.assertIn("first_trigger", first)
        self.assertNotIn("second_trigger", first)
        self.assertNotIn("first_trigger", second)
        self.assertIn("second_trigger", second)
        self.assertIn("garden", first)
        self.assertIn("garden", second)

    def test_none_zero_and_reusing_one_lora_create_separate_hook_groups(self):
        _, positive, *_ = self.compose(character_1_lora="None", character_2_strength=0)
        self.assertFalse(self.loaded)
        self.assertTrue(all("hooks" not in entry[1] for entry in positive))
        _, positive, *_ = self.compose(character_2_lora="a.safetensors")
        self.assertIsNot(positive[0][1]["hooks"], positive[1][1]["hooks"])

    def test_unmatched_lora_is_an_error_instead_of_silently_ignoring_it(self):
        self.lora.load_lora = lambda *args: {}
        with self.assertRaisesRegex(ValueError, "no matching Z-Image"):
            self.compose()

    def test_inherited_clip_hooks_are_rejected(self):
        self.args["clip"].existing_hooks = object()
        with self.assertRaisesRegex(ValueError, "CLIP already attaches"):
            self.compose()

    def test_dynamic_model_returns_static_delegate_for_native_hooks(self):
        model, *_ = self.compose(model=Model(True))
        self.assertFalse(model.is_dynamic())
        self.assertEqual(model.mode, "minvram")

    def test_invalid_latents_models_percentages_and_prompts(self):
        for value in (0, 100, math.inf, math.nan, True):
            with self.subTest(percent=value), self.assertRaises(ValueError):
                self.compose(character_1_percent=value)
        for shape in ((1, 4, 12, 16), (1, 16, 1, 12, 16), (1, 16, 11, 16)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.compose(latent={"samples": torch.zeros(shape)})
        bad = Model(); bad.model.model_config.unet_config={"image_model": "krea2"}
        with self.assertRaisesRegex(ValueError, "native latent-space Z-Image"):
            self.compose(model=bad)
        with self.assertRaises(ValueError): self.compose(character_1_prompt=" ")
        with self.assertRaises(ValueError): self.compose(direction="Diagonal")
        with self.assertRaises(ValueError): self.compose(separation="Unknown")
        self.runtime.assert_not_called()

    def test_invalid_strength_and_missing_file_are_errors(self):
        with self.assertRaises(ValueError): self.compose(character_1_strength=math.nan)
        self.paths.get_full_path_or_raise = lambda *args: (_ for _ in ()).throw(FileNotFoundError("Missing LoRA"))
        with self.assertRaises(FileNotFoundError): self.compose()

    def test_inventory_and_package_registration_need_no_other_extension(self):
        paths = types.SimpleNamespace(get_filename_list=lambda _: [])
        with patch.dict(sys.modules, {"folder_paths": paths}):
            schema = composer.ZImageTwoCharacterComposer.INPUT_TYPES()["required"]
        self.assertEqual(schema["character_1_lora"][0], ["None"])
        self.assertEqual(set(package.NODE_CLASS_MAPPINGS), {"ZImageTwoCharacterComposer"})

    def test_frontend_widget_order_matches_input_types(self):
        # composer.js restores saved values by position, so its list must be the node's own
        # widget order: required inputs first, then optional ones, skipping sockets.
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is needed to read web/composer.js")
        script = ("const vm = require('node:vm'), fs = require('node:fs');"
                  "const c = vm.createContext({ app: { registerExtension() {} } });"
                  "vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/^import .*;\\n/, ''), c);"
                  "console.log(JSON.stringify(vm.runInContext('CURRENT_ORDER', c)));")
        output = subprocess.run([node, "-e", script, str(ROOT / "web" / "composer.js")],
                                capture_output=True, text=True, check=True).stdout
        paths = types.SimpleNamespace(get_filename_list=lambda _: ["a.safetensors"])
        with patch.dict(sys.modules, {"folder_paths": paths}):
            inputs = composer.ZImageTwoCharacterComposer.INPUT_TYPES()
        widgets = []
        for section in ("required", "optional"):
            for name, spec in inputs[section].items():
                kind, options = spec[0], (spec[1] if len(spec) > 1 else {})
                if not options.get("forceInput") and (isinstance(kind, list) or kind in ("STRING", "INT", "FLOAT", "BOOLEAN")):
                    widgets.append(name)
        self.assertEqual(json.loads(output), widgets)

    def test_free_boxes_add_background_without_character_hooks(self):
        _, positive, _, masks, negative, report = self.compose(direction='Free boxes')
        self.assertEqual(len(positive), 3)
        self.assertNotIn('hooks', positive[2][1])
        self.assertEqual(masks.shape, (2, 96, 128))
        self.assertIn('Encoded branches:', report)
        self.assertTrue(all(composer.METADATA_KEY not in item[1] for item in negative))
        self.assertTrue(all(not bool(item[0].any()) for item in negative))

    def test_common_prompts_triggers_and_metadata_keep_character_roles(self):
        _, positive, _, _, _, _ = self.compose(
            base_prompt='A garden, {outfit: navy uniform}', character_1_trigger='hero_a',
            character_1_prompt='a knight in {outfit}', character_2_trigger='hero_b',
            character_2_prompt='a wizard in {outfit}')
        meta = positive[0][1][composer.METADATA_KEY]
        self.assertEqual(meta['character_prompts'], ['hero_a, a knight in navy uniform', 'hero_b, a wizard in navy uniform'])
        self.assertNotIn('hero_b', self.args['clip'].prompts[0])
        self.assertNotIn('hero_a', self.args['clip'].prompts[1])
        self.assertEqual(meta['scene_prompt'], 'A garden, navy uniform')
        self.assertEqual(meta['model_family'], 'zimage')
        self.assertEqual(meta['width'], 128)

    def test_negative_bridge_is_optional_and_used_only_for_needed_encodings(self):
        bridge = sys.modules.get(composer.__package__+'.negpip_bridge')
        if bridge is None:
            import importlib
            bridge = importlib.import_module(composer.__package__+'.negpip_bridge')
        patched = Clip()
        validate = lambda cond: None
        with patch.object(bridge, 'patch', return_value=(patched, lambda p, n: p+' NEG '+n, validate)) as call:
            self.compose(negative_prompt='watermark')
            call.assert_called_once()
            self.assertEqual(len(patched.prompts), 2)
            self.assertTrue(all('NEG watermark' in text for text in patched.prompts))
        with patch.object(bridge, 'patch', side_effect=AssertionError('should not patch')):
            self.compose()

    def test_inline_tags_load_regionally_and_do_not_enter_prompt_text(self):
        _, positive, *_ = self.compose(character_1_lora='None', character_1_prompt='<lora:a:0.4> a knight')
        self.assertEqual(positive[0][1]['hooks'].hooks[0].strength, .4)
        self.assertNotIn('<lora:', self.args['clip'].prompts[0])

    def test_example_workflow_links_and_native_loader_types(self):
        w=json.loads((ROOT / "example_workflows/two_characters.json").read_text())
        nodes={n["id"]:n for n in w["nodes"]}
        for id,src,slot,dst,inslot,kind in w["links"]:
            output=nodes[src]["outputs"][slot]; input=nodes[dst]["inputs"][inslot]
            self.assertIn(id,output["links"]); self.assertEqual(input["link"],id)
            self.assertEqual(input["type"],kind); self.assertEqual(output["type"],kind)
        self.assertFalse(any("Krea" in n["type"] for n in nodes.values()))
        self.assertEqual(next(n for n in nodes.values() if n["type"]=="CLIPLoader")["widgets_values"][1], "lumina2")

    def test_toggle_example_switches_all_four_generation_routes_together(self):
        w = json.loads((ROOT/'example_workflows/toggle_composer.json').read_text())
        nodes = {n['id']: n for n in w['nodes']}
        links = {l[0]: l for l in w['links']}
        sampler = nodes[7]
        for index, off, on in [(0, (1, 0), (5, 0)), (1, (12, 0), (5, 1)),
                               (2, (13, 0), (5, 4)), (3, (4, 0), (5, 2))]:
            switch = nodes[links[sampler['inputs'][index]['link']][1]]
            self.assertEqual(switch['type'], 'ComfySwitchNode')
            self.assertEqual(links[switch['inputs'][2]['link']][1:3], [10, 0])
            self.assertEqual(tuple(links[switch['inputs'][0]['link']][1:3]), off)
            self.assertEqual(tuple(links[switch['inputs'][1]['link']][1:3]), on)

    def test_zero_negative_drops_negpip_token_metadata_but_keeps_hooks_and_masks(self):
        tensor = torch.ones(1, 3, 4)
        mask, hooks = object(), object()
        original = [[tensor, {'cyberdelia_negpip_zimage_tokens': tensor, 'cyberdelia_negpip_zimage_strength': tensor,
                              'cyberdelia_negpip_zimage_suffix_length': 1, 'hooks': hooks, 'mask': mask,
                              composer.METADATA_KEY: {'scene_prompt': 'scene'}}]]
        zero = composer.zero_conditioning(original)
        self.assertEqual(set(zero[0][1]), {'hooks', 'mask'})
        self.assertIs(zero[0][1]['hooks'], hooks)
        self.assertIs(zero[0][1]['mask'], mask)
        self.assertFalse(bool(zero[0][0].any()))
        self.assertTrue(bool(original[0][0].all()))


if __name__ == "__main__":
    unittest.main()
