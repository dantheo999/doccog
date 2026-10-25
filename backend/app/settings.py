from dataclasses import dataclass, field
import os
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    endpoint: str = ""
    deployment: str = ""
    api_key: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.endpoint:
            return
        url = urlsplit(self.endpoint)
        azure_hosts = (".openai.azure.com", ".services.ai.azure.com", ".cognitiveservices.azure.com")
        if (url.scheme != "https" or not url.hostname or not url.hostname.endswith(azure_hosts)
                or url.username or url.password or url.query or url.fragment
                or url.path.rstrip("/") not in ("", "/openai/v1")):
            raise ValueError("Azure endpoint must be an HTTPS Azure resource URL or its /openai/v1/ URL.")

    @property
    def base_url(self) -> str:
        base = self.endpoint.rstrip("/")
        return base + "/" if base.endswith("/openai/v1") else base + "/openai/v1/"

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.deployment and self.api_key)


def load_settings() -> Settings:
    return Settings(
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip(),
        deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip(),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", "").strip(),
    )
