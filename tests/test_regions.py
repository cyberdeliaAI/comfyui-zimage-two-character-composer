import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('zimage_regions_test', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
regions = sys.modules[spec.name+'.regions']


class RegionTests(unittest.TestCase):
    def masks(self, **changes):
        options = dict(height=32, width=48, direction='Free boxes', percent=60,
                       boxes='5, 10, 40, 55; 35, 5, 60, 85', first_mask=None, second_mask=None,
                       edge_softness=0, separation='Strong')
        return regions.layout_masks(**(options | changes))[0]

    def test_free_boxes_have_exclusive_priority_and_base_only_background(self):
        masks = self.masks()
        torch.testing.assert_close(masks.sum(0), torch.ones(32, 48))
        self.assertEqual(float((masks[0] * masks[1]).sum()), 0)
        self.assertGreater(float(masks[2].sum()), 0)
        self.assertEqual(float(masks[0, 8, 18]), 1)
        self.assertEqual(float(masks[1, 8, 18]), 0)

    def test_full_split_needs_no_background_pass(self):
        masks = self.masks(direction='Left / Right')
        self.assertEqual(float(masks[2].sum()), 0)

    def test_balanced_softness_blends_predictions_but_conserves_weight(self):
        hard = self.masks()
        soft = self.masks(separation='Balanced', edge_softness=32)
        torch.testing.assert_close(soft.sum(0), torch.ones(32, 48))
        self.assertGreater(float((soft[0] * soft[1]).sum()), 0)
        torch.testing.assert_close(hard, self.masks(separation='Strong', edge_softness=64))

    def test_mask_inputs_override_boxes_and_only_first_batch_entry_is_used(self):
        mask = torch.zeros(2, 64, 96)
        mask[0, :, :32] = 1
        mask[1] = 1
        out = self.masks(first_mask=mask)
        self.assertTrue(bool((out[0, :, :16] == 1).all()))
        self.assertEqual(float(out[0, :, 16:].sum()), 0)

    def test_empty_faint_thin_and_covered_regions_are_rejected(self):
        for mask in (torch.zeros(32, 48), torch.ones(32, 48)*.2):
            with self.assertRaisesRegex(ValueError, 'area of its own'):
                self.masks(first_mask=mask)
        with self.assertRaisesRegex(ValueError, 'Character 2'):
            self.masks(boxes='0, 0, 100, 100; 20, 20, 40, 40')

    def test_invalid_mask_shapes_and_nonfinite_values_are_rejected(self):
        for mask in (torch.zeros(0, 32, 48), torch.zeros(1, 1, 32, 48), torch.full((32, 48), float('nan'))):
            with self.assertRaises(ValueError):
                self.masks(first_mask=mask)


if __name__ == '__main__':
    unittest.main()
