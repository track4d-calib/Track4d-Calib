"""Reuse identical static triangulations within a fresh calibration run."""

import ast
import base64
from collections import OrderedDict
from copy import deepcopy
from functools import wraps
import gzip
import hashlib
import inspect
import json
import textwrap

import numpy as np


def _array_key(value):
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def observation_key(observations, config):
    settings = tuple(sorted(vars(config).items())) if hasattr(config, '__dict__') else config
    parts = [repr(settings).encode()]
    for view, camera, point in observations:
        parts.append(view.encode())
        for value in (camera.K, camera.D, camera.R, camera.T, np.asarray(point, dtype=np.float64)):
            dtype, shape, data = _array_key(value)
            parts.extend((dtype.encode(), repr(shape).encode(), data))
    packed = b''.join(len(part).to_bytes(4, 'little') + part for part in parts)
    return hashlib.sha256(packed).hexdigest()


def solver_signature(function):
    """Identify the numerical solver independently of paths and docstrings."""
    function = inspect.unwrap(function)
    dependencies = [function]
    for name in ['_normalized_point', '_project', '_triangulate_normalized', 'triangulate_point_dlt']:
        if name in function.__globals__:
            dependencies.append(function.__globals__[name])
    camera = function.__globals__.get('CameraParams')
    if camera is not None:
        dependencies.append(camera.camera_center.fget)
    sources = []
    for dependency in dependencies:
        tree = ast.parse(textwrap.dedent(inspect.getsource(dependency)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.decorator_list = []
                if ast.get_docstring(node):
                    node.body = node.body[1:]
        sources.append(ast.dump(tree, include_attributes=False))
    return hashlib.sha256('\n'.join(sources).encode()).hexdigest()


def _encode_array(value):
    if not isinstance(value, np.ndarray):
        raise TypeError(type(value).__name__)
    return {'array_dtype': value.dtype.str, 'array_shape': value.shape,
            'array_bytes': base64.b64encode(value.tobytes()).decode('ascii')}


def _decode_array(value):
    if set(value) == {'array_dtype', 'array_shape', 'array_bytes'}:
        raw = base64.b64decode(value['array_bytes'], validate=True)
        return np.frombuffer(raw, dtype=value['array_dtype']).reshape(value['array_shape']).copy()
    return value


def memoize_triangulation(function, capacity=8192):
    cache = OrderedDict()
    counts = {"hits": 0, "misses": 0}

    @wraps(function)
    def triangulate(*, observations, config):
        key = observation_key(observations, config)
        if key in cache:
            counts["hits"] += 1
            cache.move_to_end(key)
            return deepcopy(cache[key])
        counts["misses"] += 1
        result = function(observations=observations, config=config)
        cache[key] = deepcopy(result)
        if len(cache) > capacity:
            cache.popitem(last=False)
        return result

    def export_cache(path):
        payload = {'format': 1, 'solver_sha256': solver_signature(function), 'entries': cache}
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump(payload, stream, default=_encode_array, allow_nan=False, separators=(',', ':'))

    triangulate.cache_info = lambda: dict(counts, size=len(cache), capacity=capacity)
    triangulate.export_cache = export_cache
    return triangulate


def load_cache(path, solver):
    """Load current-run results only when their numerical solver matches."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        payload = json.load(stream, object_hook=_decode_array)
    compatible = payload.get('format') == 1 and payload.get('solver_sha256') == solver_signature(solver)
    entries = payload['entries'] if compatible else {}
    counts = {'hits': 0, 'misses': 0, 'solver_match': compatible}

    def triangulate(*, observations, config):
        key = observation_key(observations, config)
        if key in entries:
            counts['hits'] += 1
            return deepcopy(entries[key])
        counts['misses'] += 1
        return solver(observations=observations, config=config)

    triangulate.cache_info = lambda: dict(counts)
    return triangulate
