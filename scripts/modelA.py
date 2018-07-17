import ModelAbstract

class modelA(ModelAbstract.ModelAbstract):
    RESOURCE_CONSTANT = 10
    _name = 'tesla'

    def get_name(self): return self._name

    def get_range(self,no_of_elements=1):
        return range(no_of_elements)

    def get_resource(self, resource_id=1) :
        return self.RESOURCE_CONSTANT * resource_id
