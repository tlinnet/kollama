import huggingface_hub
import requests


class _TimeoutSession(requests.Session):
    """By default Session does not use a timeout for requests unless one is provided.
    This means that requests can potentially wait forever which in our case can freeze the
    entire application."""

    def __init__(self, timeout=None):
        super().__init__()
        self.default_timeout = timeout

    def request(self, method, url, **kwargs):
        if "timeout" not in kwargs and self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout
        return super().request(method, url, **kwargs)


def _http_backend_factory() -> requests.Session:
    return _TimeoutSession(5)


huggingface_hub.configure_http_backend(_http_backend_factory)
