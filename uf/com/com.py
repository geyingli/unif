import os
import logging
import numpy as np

from ..third import tf

PACK_DIR = os.path.dirname(__file__)


class Null:
    """ A null class for keeping code compatible when hanging out. """
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def unimported_module(name, message):
    """ Returns an invalid module where error occurs only when being called. """

    class UnimportedModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(message)
    return UnimportedModule


def warning(func):
    """ A function wrapper to avoid application crash. """
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            tf.logging.warning(e)
    return wrapper


def set_verbosity(level=2):
    """ Set exposure level of detail information. """
    if level == 2:
        tf.logging.set_verbosity(tf.logging.INFO)
    elif level == 1:
        tf.logging.set_verbosity(tf.logging.WARN)
    elif level == 0:
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        raise ValueError(
          "Invalid value: %s. Pick from `0`, `1` and `2`. "
          "The larger the value, the more information will be printed." % level
        )


def set_log(log_file):
    """ Set logging file. """
    log = logging.getLogger("tensorflow")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    log.addHandler(fh)


def truncate_segments(segments, max_seq_length, truncate_method="LIFO"):
    """ Truncate sequence segments to avoid the overall length exceeds the `max_seq_length`. """
    total_seq_length = sum([len(segment) for segment in segments])
    if total_seq_length <= max_seq_length:
        return
    if truncate_method not in ("longer-FO", "FIFO", "LIFO"):
        raise ValueError("Invalid value for `truncate_method`. Pick one from `FIFO`, `LIFO` and `longer-FO`.")

    n = 0
    if truncate_method == "FIFO":
        index = 0
        while n < total_seq_length - max_seq_length:
            if not segments[index]:
                index += 1
                continue
            segments[index].pop(0)
            n += 1
    elif truncate_method == "LIFO":
        index = len(segments) - 1
        while n < total_seq_length - max_seq_length:
            if not segments[index]:
                index -= 1
                continue
            segments[index].pop()
            n += 1
    else:
        while n < total_seq_length - max_seq_length:
            max(segments, key=lambda x: len(x)).pop()
            n += 1

def transform(output_arrays, n_inputs=None):
    """ Transform raw outputs. """

    # consolidate different batches
    if isinstance(output_arrays[0], np.ndarray):
        if len(output_arrays[0].shape) == 1:    # 1D
            out = np.hstack(output_arrays)
        else:                                   # 2D/3D/...
            out = np.vstack(output_arrays)
        return out[:n_inputs] if n_inputs else out

    # flatten
    elif isinstance(output_arrays[0], list):
        out = [item for output_array in output_arrays for item in output_array]
        return out[:n_inputs] if n_inputs else out

    else:
        return output_arrays
