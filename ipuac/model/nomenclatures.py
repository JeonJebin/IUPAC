from enum import Enum


class Alkyl(Enum):
    ISOPROPYL = ('isopropyl', 3, 1, 2)
    ISOBUTYL = ('isobutyl', 4, 2, 2)
    SECBUTYL = ('sec-butyl', 4, 1, 2)
    TERTBUTYL = ('tert-butyl', 4, 1, 3)
    ISOPENTYL = ('isopentyl', 5, 3, 2)
    NEOPENTYL = ('neopentyl', 5, 2, 3)
    TERTPENTYL = ('tert-pentyl', 5, 1, 3)
    ISOHEXYL = ('isohexyl', 6, 4, 2)

    METHYL = ('methyl', 1, 0, 0)
    ETHYL = ('ethyl', 2, 0, 0)
    PROPHYL = ('prophyl', 3, 0, 0)
    BUTYL = ('butyl', 4, 0, 0)
    PENTYL = ('pentyl', 5, 0, 0)
    HEXYL = ('hexyl', 6, 0, 0)
    HEPTYL = ('heptyl', 7, 0, 0)
    OCTYL = ('octyl', 8, 0, 0)
    NONYL = ('nonyl', 9, 0, 0)
    DECYL = ('decyl', 10, 0, 0)
    UNDECYL = ('undecyl', 11, 0, 0)
    DODECYL = ('dodecyl', 12, 0, 0)

    NONE = ('', 0, 0, 0)

    def __init__(self, aname, vertax_number, start_branch, branch_number):
        self.aname = aname
        self.vertax_number = vertax_number
        self.start_branch = start_branch
        self.branch_number = branch_number

    @staticmethod
    def get_name(compare_info):
        for name in Alkyl:
            if compare_info == (name.vertax_number, name.start_branch, name.branch_number):
                return name.aname
        return Alkyl.NONE.aname


class Halogen(Enum):
    FLUORO = ('fluoro', 'f')
    CHLORO = ('chloro', 'cl')
    BROMO = ('bromo', 'br')
    IDOD = ('iodo', 'i')
    NONE = ('', '')

    def __init__(self, aname, element_name):
        self.aname = aname
        self.element_name = element_name

    @staticmethod
    def get_name(element_name):
        for name in Halogen:
            if element_name == name.element_name:
                return name.aname
        return Halogen.NONE.aname


class ParentChainName(Enum):
    METHANE = ('methane', 1)
    ETHANE = ('ethane', 2)
    PROPANE = ('propane', 3)
    BUTANE = ('butane', 4)
    PENTANE = ('pentane', 5)
    HEXANE = ('hexane', 6)
    HEPTANE = ('heptane', 7)
    OCTANE = ('octane', 8)
    NONANE = ('nonane', 9)
    DECANE = ('decane', 10)
    UNDECANE = ('undecane', 11)
    DODECANE = ('dodecane', 12)
    TRIDECANE = ('tridecane', 13)
    BUTADECANE = ('butadecane', 14)
    PENTADECANE = ('pentadecane', 15)
    HEXADECANE = ('hexadecane', 16)
    HEPTADECANE = ('heptadecane', 17)
    OCDTADECANE = ('octadecane', 18)
    NONAOANE = ('nonadecane', 19)
    ICOSANE = ('icosane', 20)
    HENICOSANE = ('henicosane', 21)
    DOCOSANE = ('docosane', 22)
    NONE = ('', 0)

    def __init__(self, aname, chain_length):
        self.aname = aname
        self.chain_length = chain_length

    @staticmethod
    def get_name(chain_length):
        for name in ParentChainName:
            if chain_length == name.chain_length:
                return name.aname
        return ParentChainName.NONE.aname


class InfixNumber(Enum):
    DI = ('di', 2)
    TRI = ('tri', 3)
    TETRA = ('tetra', 4)
    PENTA = ('penta', 5)
    HEXA = ('hexa', 6)
    HEPTA = ('hepta', 7)
    OCTA = ('octa', 8)
    NONA = ('nona', 9)
    DECA = ('deca', 10)
    UNDECA = ('undeca', 11)
    DODECA = ('dodeca', 12)
    TRIDECA = ('trideca', 13)
    BUTADECA = ('butadeca', 14)
    PENTADECA = ('pentadeca', 15)
    HEXADECA = ('hexadeca', 16)
    HEPTADECA = ('heptadeca', 17)
    OCDTADECA = ('octadeca', 18)
    NONAOA = ('nonadeca', 19)
    ICOSA = ('icosa', 20)
    HENICOSA = ('henicosa', 21)
    DOCOSA = ('docosa', 22)
    NONE = ('', 0)

    def __init__(self, aname, chain_length):
        self.aname = aname
        self.chain_length = chain_length

    @staticmethod
    def get_name(chain_length):
        for name in InfixNumber:
            if chain_length == name.chain_length:
                return name.aname
        return InfixNumber.NONE.aname