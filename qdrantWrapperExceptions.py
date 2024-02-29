class CollectionNameNotProvidedException(Exception):
    """Raised when the collection name is not provided."""
    pass

class CollectionAlreadyExistsException(Exception):
    """Raised when the collection already exists."""
    pass