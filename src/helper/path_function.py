import os
import pathlib
import re


def remove_illegal_char(path: str, replace_with=""):
    new_str = re.sub("[^0-9a-zA-Z.!@#$(){}\[\]\-_,' ]*", replace_with, path.strip())

    if len(new_str) == 0:
        return ""

    while ".." in new_str:
        new_str = new_str.replace("..", ".")

    if len(new_str) == 0:
        return ""

    if new_str[-1] == ".":
        new_str = new_str[:-1]
    return new_str


def get_relative_path(this, path):
    return os.path.normpath(os.path.abspath(os.path.join(
        pathlib.Path(this).parent.absolute(), path)
    ))


def path_join(*args):
    args = list(args)
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        path = args.pop(0)
        for i in args:
            path = os.path.normpath(os.path.abspath(os.path.join(path, i)))
        return path