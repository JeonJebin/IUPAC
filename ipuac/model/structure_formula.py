

class StructureFormula:
    def __init__(self, parent_chain, functional_groups):
        self.parent_chain = parent_chain
        self.functional_groups = functional_groups

    @staticmethod
    def create(image, body, combine_info):
        parent_chain = body.find_parent_chain(image, combine_info)
        functional_groups = parent_chain.find_functional_groups(body, combine_info)
        return StructureFormula(parent_chain, functional_groups)

    def get_name(self):
        return self.functional_groups.get_name() + self.parent_chain.get_name()
