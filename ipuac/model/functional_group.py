from ipuac.model.nomenclatures import *


class FunctionalGroup:
    def __init__(self, positions, count, suffix):
        self.__positions = positions
        self.__count = count
        self.__suffix = suffix

    @property
    def element(self):
        return self.__count

    @property
    def position(self):
        return self.__positions

    def __add__(self, other):
        return FunctionalGroup(self.__positions + other.__positions, self.__count + 1, self.__suffix)

    def get_name(self):
        prefix = ','.join(map(str, self.__positions))
        infix = InfixNumber.get_name(len(self.__positions))
        return "{}-{}{}".format(prefix, infix, self.__suffix)

    def is_equal_suffix(self, suffix):
        return self.__suffix == suffix
