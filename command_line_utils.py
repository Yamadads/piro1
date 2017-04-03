import re
import os


def represents_int(s):
    return re.match(r"[-+]?\d+$", s) is not None


def check_command_line_arguments(args):
    if len(args) != 3:
        print("Wrong number of arguments. Exactly two arguments are needed.")
        return False
    if not represents_int(args[2]):
        print ("Second argument must be integer value.")
        return False
    if not os.path.isdir(args[1]):
        print ("The given path does not exist.")
        return False
    return True
