import os
import re

from ..third import tf


def get_checkpoint_path(path):
    """ If detected no checkpoint file, return None. """

    # get directory
    dir_name = path if os.path.isdir(path) else os.path.dirname(path)
    if not dir_name:
        dir_name = "."

    # get file
    file_name = ""
    if not os.path.isdir(path):
        file_name = path.strip("/").split("/")[-1]

        # find checkpoint
        if os.path.isfile(f"{dir_name}/{file_name}.index"):
            return f"{dir_name}/{file_name}"

        # stop to avoid error
        return None

    # get file from record file
    if os.path.exists(f"{dir_name}/checkpoint"):
        with open(f"{dir_name}/checkpoint") as f:
            line = f.readline()
        try:
            file = re.findall("model_checkpoint_path: \"(.+?)\"", line)[0]
            if os.path.exists(f"{dir_name}/{file}.index"):
                return f"{dir_name}/{file}"
        except IndexError:
            pass

    # find file with largest step
    files = []
    for file in os.listdir(dir_name):
        prefix = re.findall("(.+?).index", file)
        if prefix:
            step = 0
            prefix = prefix[0]
            try:
                step = int(prefix.split("-")[-1])
            except:
                pass
            files.append((step, file))
    if files:
        files.sort(key=lambda x: x[0], reverse=True)
        file = files[0][1].replace(".index", "")
        return f"{dir_name}/{file}"

    # find no checkpoint
    return None


def get_assignment_map(checkpoint_file, variables, continual=False, show_matched=False):
    """ Carefully designed so as to fulfil any personalized needs. """
    assignment_map = {}

    # read local variables
    name_to_variable = {}
    for var in variables:
        name = var.name
        res = re.match("^(.*):\\d+$", name)
        if res is not None:
            name = res.group(1)
        if not continual:
            if "global_step" in name \
                    or "/adam" in name \
                    or "/Adam" in name \
                    or "/lamb" in name:
                continue
        name_to_variable[name] = var

    # read checkpoint variables
    init_vars = tf.train.list_variables(checkpoint_file)
    inited_vars = {}
    for name_shape in init_vars:
        (from_name, from_shape) = (name_shape[0], name_shape[1])

        to_name = from_name
        if to_name not in name_to_variable or \
                name_to_variable[to_name].shape.as_list() != from_shape:
            if show_matched:
                tf.logging.info("checkpoint_file contains <%s>", from_name)
            continue
        if show_matched:
            tf.logging.info("checkpoint_file contains <%s>, matched", from_name)
        assignment_map[from_name] = name_to_variable[to_name]
        inited_vars[to_name] = 1

    # further feedback
    uninited_vars = {}
    for var in variables:
        if var.name[:-2] not in inited_vars:
            if var.name[:-2].endswith("_m") or var.name[:-2].endswith("_v"):
                continue
            if show_matched:
                tf.logging.info("unmatched parameter %s", var)
            uninited_vars[var.name[:-2]] = var
    return (assignment_map, uninited_vars)


def list_variables(checkpoint):
    checkpoint_path = get_checkpoint_path(checkpoint)
    if not checkpoint_path:
        raise ValueError(
            "Checkpoint file \"%s\" does not exist. "
            "Make sure you pass correct value to "
            "`checkpoint`." % checkpoint
        )
    return tf.train.list_variables(checkpoint_path)
