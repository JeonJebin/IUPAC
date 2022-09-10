class FunctionalGroups:
    def __init__(self, functional_groups):
        self.functional_groups = functional_groups

    def get_name(self):
        name = []
        for functional_group in self.functional_groups:
            name.append(functional_group.get_name())
        return '-'.join(name)
