from .functional_group import FunctionalGroup
from .functional_groups import FunctionalGroups
from .nomenclatures import ParentChainName


class ParentChain:
    def __init__(self, path):
        self.__length, self.__path = path

    def get_name(self):
        return ParentChainName.get_name(self.__length)

    def find_functional_groups(self, body, combine_info):
        candidates = body.get_functional_group_candidates(self.__path, combine_info)
        candidates.sort(key=lambda x: (x[1], x[0]))

        (position, name) = candidates[0]
        functional_groups = [FunctionalGroup([position], 1, name)]
        for i in range(1, len(candidates)):
            if functional_groups[-1].is_equal_suffix(candidates[i][1]):
                functional_groups[-1] += FunctionalGroup([candidates[i][0]], 1, candidates[i][1])
            else:
                functional_groups.append(FunctionalGroup([candidates[i][0]], 1, candidates[i][1]))

        return FunctionalGroups(functional_groups)
