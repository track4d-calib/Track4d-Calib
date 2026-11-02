"""Reuse deterministic image features within one RoMa inference session."""

from collections import OrderedDict
from functools import wraps
from pathlib import Path


def _copy(value):
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    return value.detach().clone()


def cache_image_features(model, capacity=32):
    """Cache features by immutable file identity and actual input tensor shape.

    Model weights and preprocessing must remain fixed for this session. Each
    return owns its tensors, so pairwise matching cannot mutate cached features.
    A model instance supports one active match call at a time.
    """
    if capacity < 1:
        raise ValueError("feature cache capacity must be positive")
    active = None
    stats = {"hits": 0, "misses": 0, "image_hits": 0, "image_misses": 0}
    counters = {}

    def identity(value):
        path = Path(value).resolve()
        stat = path.stat()
        return (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)

    images = OrderedDict()
    original_load = model._load_image

    @wraps(original_load)
    def load_image(value):
        if not isinstance(value, (str, Path)):
            return original_load(value)
        key = identity(value)
        if key in images:
            stats["image_hits"] += 1
            images.move_to_end(key)
            return _copy(images[key])
        image = original_load(value)
        images[key] = _copy(image)
        stats["image_misses"] += 1
        while len(images) > capacity:
            images.popitem(last=False)
        return image

    model._load_image = load_image
    for name in ("f", "refiner_features"):
        module = getattr(model, name)
        original = module.forward
        cache = OrderedDict()

        def cached(tensor, *args, _name=name, _original=original, _cache=cache, **kwargs):
            if active is None:
                return _original(tensor, *args, **kwargs)
            index = counters[_name]
            counters[_name] += 1
            key = (active[index % 2], tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
            if key in _cache:
                stats["hits"] += 1
                _cache.move_to_end(key)
                return _copy(_cache[key])
            result = _original(tensor, *args, **kwargs)
            _cache[key] = _copy(result)
            stats["misses"] += 1
            while len(_cache) > capacity:
                _cache.popitem(last=False)
            return result

        module.forward = cached
    original_match = model.match

    @wraps(original_match)
    def match(first, second):
        nonlocal active
        if active is not None:
            raise RuntimeError("concurrent matching on a cached model is unsupported")
        if not all(isinstance(path, (str, Path)) for path in (first, second)):
            return original_match(first, second)
        active = tuple(identity(value) for value in (first, second))
        counters.update(f=0, refiner_features=0)
        try:
            return original_match(first, second)
        finally:
            active = None

    model.match = match
    return stats
