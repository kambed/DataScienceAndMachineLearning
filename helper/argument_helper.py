from enum import Enum
import json


class ArgumentHelper:
    @staticmethod
    def get_enum_argument(arg_name, enum_class):
        if not issubclass(enum_class, Enum):
            raise ValueError("Second argument must be an Enum class")

        value = ArgumentHelper.get_argument(arg_name)
        try:
            return enum_class[value]
        except KeyError:
            raise ValueError(f"Invalid value {value} for argument {arg_name}")

    @staticmethod
    def get_int_argument(arg_name):
        value = ArgumentHelper.get_argument(arg_name)
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid value {value} for argument {arg_name}")

    @staticmethod
    def get_argument(arg_name):
        with open("resources/Algorithm_Arguments.json") as f:
            arguments = json.load(f)
            if arg_name not in arguments:
                raise ValueError(f"Missing argument {arg_name} in Algorithm_Arguments.json")

            print(f"{arg_name}: {arguments[arg_name]}")

            return arguments[arg_name]
