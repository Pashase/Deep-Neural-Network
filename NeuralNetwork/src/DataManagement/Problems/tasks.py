import typing as tp
import enum


@enum.unique
class ConsiderTasks(enum.Enum):
    DigitRecognitionTask = 'DigitRecognition'
    CatNonCatTask = 'CatNonCat'

    @staticmethod
    def values() -> tp.List:
        return [task.value for task in ConsiderTasks]
