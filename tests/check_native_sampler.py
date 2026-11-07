"""CPU integration with real ComfyUI hooks and its native sampling loop.

Requires PyTorch and NumPy. Pass trusted ComfyUI source files:
python tests/check_native_sampler.py /path/comfy/hooks.py /path/comfy/samplers.py /path/node_helpers.py

LoRA file parsing, CLIP and model execution are test doubles. The two LoRAs are
actual rank-one tensor projections; hook registration/attachment and sampler
grouping/blending use the supplied official code. This does not run Z-Image.
"""

import ast
import collections
import importlib.util
import logging
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch


def extract_functions(path, names, scope):
    tree = ast.parse(Path(path).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in tree.body} == set(names)
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(tree), path, "exec"), scope)


def main(hooks_path, samplers_path, helpers_path):
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.lora = types.ModuleType("comfy.lora")
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.patcher_extension = types.ModuleType("comfy.patcher_extension")
    comfy.utils = types.ModuleType("comfy.utils")
    comfy.lora.model_lora_keys_unet = lambda model, mapping: {"adapter": "diffusion_model.projection.weight"}
    comfy.lora.load_lora = lambda state, mapping: {"diffusion_model.projection.weight": state}
    states = {
        "a.pt": {"down": torch.tensor([[1., 0.]]), "up": torch.tensor([[2.], [0.]])},
        "b.pt": {"down": torch.tensor([[0., 1.]]), "up": torch.tensor([[0.], [3.]])},
    }
    comfy.utils.load_torch_file = lambda path, safe_load: states[path]
    helpers = types.ModuleType("node_helpers")
    extract_functions(helpers_path, {"conditioning_set_values"}, helpers.__dict__)
    modules = {
        "comfy": comfy, "comfy.lora": comfy.lora, "comfy.model_management": comfy.model_management,
        "comfy.patcher_extension": comfy.patcher_extension, "comfy.utils": comfy.utils,
        "node_helpers": helpers,
        "folder_paths": types.SimpleNamespace(get_full_path_or_raise=lambda kind, name: name, get_filename_list=lambda kind: ["a.pt", "b.pt"]),
    }
    with patch.dict(sys.modules, modules):
        hook_spec = importlib.util.spec_from_file_location("comfy.hooks", hooks_path)
        hooks = importlib.util.module_from_spec(hook_spec)
        sys.modules["comfy.hooks"] = hooks
        comfy.hooks = hooks
        hook_spec.loader.exec_module(hooks)
        package_root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("zcomposer_integration", package_root / "__init__.py",
                                                     submodule_search_locations=[str(package_root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        composer = sys.modules[f"{spec.name}.composer"]
        scope = dict(torch=torch, comfy=comfy, collections=collections)
        names = {"_calc_cond_batch", "get_area_and_mult", "add_area_dims", "cond_equal_size", "can_concat_cond", "cond_cat"}
        extract_functions(samplers_path, names, scope)

        class ToyPatcher:
            def __init__(self):
                self.model = types.SimpleNamespace(model_config=types.SimpleNamespace(unet_config={"image_model":"lumina2", "dim":3840}))
                self.specs = {}
                self.active = None
                self.history = []
            def clone(self):
                return ToyPatcher()
            def is_dynamic(self):
                return False
            def set_hook_mode(self, mode):
                self.mode = mode
            def register_all_hook_patches(self, group, target, options=None, registered=None):
                registered = registered if registered is not None else hooks.HookGroup()
                for hook in group.get_type(hooks.EnumHookType.Weight):
                    hook.add_hook_patches(self, options, target, registered)
                return registered
            def add_hook_patches(self, hook, patches, strength_patch=1.0):
                self.specs[hook.hook_ref] = (patches, strength_patch)
            def prepare_hook_patches_current_keyframe(self, *args):
                pass
            def prepare_state(self, *args):
                pass
            def get_free_memory(self, *args):
                return 10**12
            def apply_hooks(self, hooks=None):
                self.active = hooks
                self.history.append(hooks)
                return {}

        class Clip:
            def tokenize(self, text):
                return text
            def encode_from_tokens_scheduled(self, tokens):
                return [[torch.zeros(1, 2, 4), {}]]

        class ToyDiffusion:
            def __init__(self, patcher):
                self.current_patcher = patcher
            def memory_required(self, *args, **kwargs):
                return 1
            def apply_model(self, x, timestep, **kwargs):
                delta = torch.zeros(2, 2)
                if self.current_patcher.active is not None:
                    for hook in self.current_patcher.active.hooks:
                        patches, strength = self.current_patcher.specs[hook.hook_ref]
                        state = patches["diffusion_model.projection.weight"]
                        delta += strength * (state["up"] @ state["down"])
                return torch.einsum("ij,bjhw->bihw", torch.eye(2) + delta, x)

        cases = 0
        layouts = [(d, m, p, 0, False) for d in composer.DIRECTIONS[:2] for m in composer.SEPARATION for p in range(1, 100)]
        layouts += [("Free boxes", m, 60, soft, False) for m in composer.SEPARATION for soft in (0, 32)]
        layouts += [("Left / Right", "Strong", 60, 0, True)]
        for direction, mode, percent, softness, inline in layouts:
            source_model = ToyPatcher()
            latent = {"samples": torch.zeros(2, 16, 8, 16)}
            patched, positive, returned_latent, previews, negative, report = composer.ZImageTwoCharacterComposer().compose(
                model=source_model, clip=Clip(), latent=latent, direction=direction, character_1_percent=percent,
                base_prompt="Shared scene", character_1_prompt="a knight" + (" <lora:b:0.4>" if inline else ""), character_1_lora="a.pt", character_1_strength=0.75,
                character_2_prompt="a wizard", character_2_lora="b.pt", character_2_strength=1.25, separation=mode, edge_softness=softness,
            )
            assert source_model.specs == {}
            assert returned_latent is latent
            groups = [extra.get("hooks") for _, extra in positive]
            combined = hooks.HookGroup.combine_all_hooks(groups)
            registered = patched.register_all_hook_patches(combined, hooks.create_target_dict(hooks.EnumWeightTarget.Model))
            assert len(registered) == (3 if inline else 2)
            conditions = [{**extra, "model_conds": {}, "uuid": str(i)} for i, (_, extra) in enumerate(positive)]
            x = torch.ones(2, 2, 8, 16)
            prediction = scope["_calc_cond_batch"](ToyDiffusion(patched), [conditions], x, torch.tensor([1.]), {})[0]
            first, second = [extra["mask"] for _, extra in positive[:2]]
            expected = x.clone()
            expected[:, 0] += 1.5 * first
            expected[:, 1] += 3.75 * second
            if inline:
                expected[:, 1] += 1.2 * first
            torch.testing.assert_close(prediction, expected)
            assert patched.history == groups
            assert previews.shape == (2, 64, 128)
            cases += 1
        print(f"Native ComfyUI hooks + sampler CPU test: {cases} layouts passed; rank-one LoRA effects stay in their own regional outputs.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
