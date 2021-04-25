import typing as tp
import enum


@enum.unique
class Label(enum.Enum):
    W = 'W'
    b = 'b'
    Z = 'Z'
    A = 'A'

    @staticmethod
    def values() -> tp.Generator:
        return (label.value for label in Label)
