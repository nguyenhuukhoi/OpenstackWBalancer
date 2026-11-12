"""One dotenv parser; loading configuration never changes process environment."""
import os
from contextvars import ContextVar
from functools import wraps

from dotenv import dotenv_values

_environment = ContextVar("configuration_environment", default=None)


def read_settings(path):
    return {key: value for key, value in dotenv_values(path).items() if value is not None} if path else {}


def getenv(name, default=None):
    values = _environment.get()
    return (os.environ if values is None else values).get(name, default)


def configuration_scope(function):
    @wraps(function)
    def wrapped(cls, filename=None):
        token = _environment.set({**read_settings(filename), **os.environ})
        try:
            return function(cls, filename)
        finally:
            _environment.reset(token)
    return wrapped
