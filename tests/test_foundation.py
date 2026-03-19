"""Smoke tests for foundation modules."""


def test_config_loads():
    from adam.config import get_config
    config = get_config()
    assert config is not None


def test_config_has_database():
    from adam.config import get_config
    config = get_config()
    assert config.database is not None
    assert config.database.url is not None


def test_config_has_sections():
    from adam.config import get_config, UnifiedConfig
    config = get_config()
    assert isinstance(config, UnifiedConfig)
    assert config.llm is not None
    assert config.memory is not None
    assert config.web is not None


def test_database_engine():
    from adam.database import get_engine
    engine = get_engine()
    assert engine is not None


def test_database_base():
    from adam.database import Base
    assert Base is not None
    assert hasattr(Base, 'metadata')


def test_errors_importable():
    from adam.errors import ADAMError, MemoryError, StorageError
    assert issubclass(MemoryError, ADAMError)
    assert issubclass(StorageError, MemoryError)


def test_errors_hierarchy():
    from adam.errors import (
        ADAMError, MemoryError, StorageError, RetrievalError,
        NetworkError, LLMError, ConfigurationError
    )
    assert issubclass(LLMError, NetworkError)
    assert issubclass(NetworkError, ADAMError)
    assert issubclass(ConfigurationError, ADAMError)
    assert issubclass(RetrievalError, MemoryError)


def test_adam_system_importable():
    from adam.core.app import ADAMSystem
    assert ADAMSystem is not None


def test_adam_version():
    import adam
    assert adam.__version__ == "4.0.0"
