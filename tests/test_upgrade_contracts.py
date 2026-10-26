"""Offline migration contracts; no SAM imports, torch, weights, network or GPU.

AST extraction executes the actual upstream routing functions with stub builders.
It deliberately does NOT validate imports, tensors, kernels, or model inference.
"""

import ast
import gzip
import json
from pathlib import Path
from typing import Optional
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def isolated_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


class UpgradeContracts(unittest.TestCase):
    def test_all_python_sources_compile(self):
        paths = list((ROOT / 'sam3').rglob('*.py')) + list((ROOT / 'scripts').rglob('*.py'))
        self.assertGreater(len(paths), 100)
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_bytes(), str(path), 'exec')

    def test_notebooks_are_valid_and_code_cells_compile(self):
        paths = list((ROOT / 'examples').glob('*.ipynb'))
        self.assertIn(ROOT / 'examples/sam3.1_video_predictor_example.ipynb', paths)
        # IPython magics/shell escapes are valid notebook syntax, not Python.
        for path in paths:
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook['nbformat'], 4)
            # Other upstream notebooks include intentionally incomplete path templates.
            if path.name != 'sam3.1_video_predictor_example.ipynb':
                continue
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    if any(line.lstrip().startswith(('%', '!', '?')) for line in source.splitlines()):
                        continue
                    compile(source, f'{path.name}:cell-{index}', 'exec')

    def test_packaged_vocabulary_is_readable(self):
        with gzip.open(ROOT / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz', 'rt') as stream:
            self.assertGreater(len(stream.read().splitlines()), 10000)
        import tomllib
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertIn('assets/*.txt.gz', project['tool']['setuptools']['package-data']['sam3'])

    def test_version_dispatch_and_argument_forwarding(self):
        old, new = Mock(return_value='old'), Mock(return_value='new')
        fn = isolated_function('sam3/model_builder.py', 'build_sam3_predictor', {
            'Optional': Optional,
            'build_sam3_video_predictor': old,
            'build_sam3_multiplex_video_predictor': new,
        })
        self.assertEqual(fn(checkpoint_path='/local/new.pt', max_num_objects=128), 'new')
        self.assertEqual(new.call_args.kwargs['max_num_objects'], 128)
        self.assertEqual(new.call_args.kwargs['multiplex_count'], 16)
        self.assertEqual(new.call_args.kwargs['checkpoint_path'], '/local/new.pt')
        self.assertEqual(fn(version='sam3', checkpoint_path='/local/old.pt'), 'old')
        self.assertNotIn('multiplex_count', old.call_args.kwargs)
        with self.assertRaises(ValueError):
            fn(version='unsupported')
        self.assertEqual(old.call_count, 1)
        self.assertEqual(new.call_count, 1)

    def test_checkpoint_repositories_and_filenames(self):
        download = Mock(return_value='/fake/checkpoint')
        fn = isolated_function('sam3/model_builder.py', 'download_ckpt_from_hf', {'hf_hub_download': download})
        fn(version='sam3.1')
        self.assertEqual(download.call_args.kwargs, {'repo_id': 'facebook/sam3.1', 'filename': 'sam3.1_multiplex.pt'})
        fn(version='sam3')
        self.assertEqual(download.call_args.kwargs, {'repo_id': 'facebook/sam3', 'filename': 'sam3.pt'})

    def test_request_dispatch_preserves_pruning_semantics(self):
        fn = isolated_function('sam3/model/sam3_base_predictor.py', 'handle_request', {}, 'Sam3BasePredictor')
        predictor = Mock()
        request = {'type': 'remove_object', 'session_id': 's', 'obj_id': 7}
        fn(predictor, request)
        self.assertTrue(predictor.remove_object.call_args.kwargs['is_user_action'])
        fn(predictor, dict(request, is_user_action=False))
        self.assertFalse(predictor.remove_object.call_args.kwargs['is_user_action'])
        with self.assertRaises(RuntimeError):
            fn(predictor, {'type': 'unknown'})

    def test_stream_dispatch_and_cache_eviction(self):
        fn = isolated_function('sam3/model/sam3_base_predictor.py', 'handle_stream_request', {}, 'Sam3BasePredictor')
        predictor = Mock(default_output_prob_thresh=0.5)
        predictor.propagate_in_video.return_value = iter([{'frame_index': 0, 'outputs': {}}])
        result = list(fn(predictor, {'type': 'propagate_in_video', 'session_id': 's', 'evict_cached_frame_outputs': True}))
        self.assertEqual(result[0]['frame_index'], 0)
        self.assertTrue(predictor.propagate_in_video.call_args.kwargs['evict_cached_frame_outputs'])
        evict = isolated_function('sam3/model/sam3_base_predictor.py', '_evict_cached_frame_output', {})
        state = {'cached_frame_outputs': {0: 'mask', 1: 'mask'}}
        evict(state, 0, False)
        self.assertIn(0, state['cached_frame_outputs'])
        evict(state, 0, True)
        evict({}, 0, True)
        self.assertEqual(state['cached_frame_outputs'], {1: 'mask'})


if __name__ == '__main__':
    unittest.main()
