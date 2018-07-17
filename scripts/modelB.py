import ModelAbstract, Utils

class modelB(ModelAbstract.ModelAbstract, Utils.Utils):
    RESOURCE_CONSTANT = 100
    _name = 'chevy'

    def get_name(self): return self._name

    def get_range(self,no_of_elements=1):
        return range(no_of_elements*2)

    def get_resource(self, resource_id=1) :
        return self.RESOURCE_CONSTANT * resource_id
