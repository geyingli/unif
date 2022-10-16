import os
import re
import json
import collections

from .. import apps
from ..third import tf


def restore(key, from_file="./.unif", **kwargs):
    """ Load model from configurations saved in local file.

    Args:
        key: string. Unique name of configuration to load.
        from_file: string. The path of configuration file.
    """
    tf.logging.info("Loading model `%s` from %s" % (key, from_file))

    if not os.path.exists(from_file):
        raise ValueError("No file found with `%s`." % from_file)
    from_fp = open(from_file, encoding="utf-8")
    from_json = json.load(from_fp)
    from_fp.close()

    if key not in from_json.keys():
        raise ValueError("No key `%s`." % key)
    model = from_json[key]["model"]
    args = collections.OrderedDict()

    # unif >= beta v2.1.35
    if "__init__" in from_json[key]:
        zips = from_json[key]["__init__"].items()
    # unif < beta v2.1.35
    elif "keys" in from_json[key]:
        zips = zip(from_json[key]["keys"], from_json[key]["values"])
    else:
        raise ValueError("Wrong format.")

    from_dir = os.path.dirname(from_file)
    if from_dir == "":
        from_dir = "."
    for arg, value in zips:

        # convert from relative path
        if arg == "init_checkpoint" or arg.endswith("_dir") or arg.endswith("_file"):
            if isinstance(value, str) and not value.startswith("/"):
                value = get_simplified_path(from_dir + "/" + value)

        if arg in kwargs:
            value = kwargs[arg]
        args[arg] = value
    return apps.__dict__[model](**args)


def load(key, cache_file="./.cache", **kwargs):
    """ Load model from configurations saved in cache file.

    NOTE: This function is deprecated and not upgraded, just retained for compatibility "
    "with older versions. Try `.restore()` instead.
    """
    return restore(key, from_file=cache_file, **kwargs)


def get_init_values(model):
    values = []
    for arg in model.__class__.__init__.__code__.co_varnames[1:]:
        try:
            value = model.__getattribute__(arg)
        except Exception:
            value = model.__init_args__[arg]
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
