import knime.extension as knext

class AIPortObjectSpec(knext.PortObjectSpec):

    def serialize(self) -> dict:
        return {}
    
    @classmethod
    def deserialize(cls, data: dict):
        return cls()

    def validate_context(self, ctx: knext.ConfigurationContext):
        """Validates that the current context allows execution e.g.
        by checking if necessary credentials are present"""