import os
import re
import json
import collections

from ..thirdparty import tf
from .. import application


def load(code, cache_file="./.cache", **kwargs):
    """ Load model from configurations saved in cache file.

    Args:
        code: string. Unique name of configuration to load.
        cache_file: string. The path of cache file.
    Returns:
        None
    """
    tf.logging.info("Loading model `%s` from %s" % (code, cache_file))

    if not os.path.exists(cache_file):
        raise ValueError("No cache file found with `%s`." % cache_file)
    cache_fp = open(cache_file, encoding="utf-8")
    cache_json = json.load(cache_fp)
    cache_fp.close()

    if code not in cache_json.keys():
        raise ValueError(
            "No cached configs found with code `%s`." % code)
    if "model" not in cache_json[code]:
        raise ValueError(
            "No model assigned. Try `uf.XXX.load()` instead.")
    model = cache_json[code]["model"]
    args = collections.OrderedDict()

    # unif >= beta v2.1.35
    if "__init__" in cache_json[code]:
        zips = cache_json[code]["__init__"].items()
    # unif < beta v2.1.35
    elif "keys" in cache_json[code]:
        zips = zip(cache_json[code]["keys"], cache_json[code]["values"])
    else:
        raise ValueError("Wrong format of cache file.")

    cache_dir = os.path.dirname(cache_file)
    if cache_dir == "":
        cache_dir = "."
    for key, value in zips:

        # convert from relative path
        if key == "init_checkpoint" or key.endswith("_dir") or \
                key.endswith("_file"):
            if isinstance(value, str) and not value.startswith("/"):
                value = get_simplified_path(
                    cache_dir + "/" + value)

        if key in kwargs:
            value = kwargs[key]
        args[key] = value
    return application.__dict__[model](**args)


def get_init_values(model):
    values = []
    for key in model.__class__.__init__.__code__.co_varnames[1:]:
        try:
            value = model.__getattribute__(key)
        except Exception:
            value = model.__init_args__[key]
        values.append(value)
    return values


def get_relative_path(source, target):
    source = source.replace("\\", "/")
    target = target.replace("\\", "/")

    if source.startswith("/"):
        raise ValueError("Not a relative path: %s." % source)
    if target.startswith("/"):
        raise ValueError("Not a relative path: %s." % target)

    output = get_reverse_path(source) + "/" + target
    output = get_simplified_path(output)
    return output


def get_simplified_path(path):
    path = path.replace("\\", "/")
    while True:
        res = re.findall("[^/]+/[.][.]/", path)
        res = [item for item in res if item != "../../" and item != "./../"]
        if res:
            path = path.replace(res[0], "")
        else:
            return path.replace("/./", "/")


def get_reverse_path(path):
    path = path.replace("\\", "/")

    if path.startswith("/"):
        raise ValueError("Not a relative path.")

    output = ""

    if os.path.isdir(path):
        if path.endswith("/"):
            path = path[:-1]
    else:
        path = os.path.dirname(path)

    if path == "":
        return "."

    cwd = os.getcwd()
    for seg in path.split("/"):
        if seg == ".":
            pass
        elif seg == "..":
            output = "/" + cwd.split("/")[-1] + output
            cwd = os.path.dirname(cwd)
        else:
            output = "/.." + output
            cwd += "/" + seg

    output = output[1:]

    if output == "":
        return "."

    return output
