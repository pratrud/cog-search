from logging import log
import os
from typing import Union
from pathlib import Path
from dotenv import load_dotenv


def load_env():

    log.info("Loading environmental variables.")
    environment = os.environ.get("LOCAL_ENV")

    if Path("./azure/azure.env").exists():
        log.info("loading environment: azure.env")
        env_path = Path(".") / "azure/azure.env"

    else:
        log.info("loading environment: dev.env")
        env_path = Path(".") / "config/dev.env"

    if env_path:
        if os.path.exists(env_path):
            load_dotenv(env_path, verbose=True)
        else:
            raise RuntimeError("No environment found. shutting down application")


load_env()

class AzureSearchConfig(object):
    _azure_search_endpoint: str
    _azure_search_key: str


    def __init__(self):
        self._azure_search_endpoint = os.environ.get("ACS_ENDPOINT")
        self._azure_search_key = os.environ.get("ACS_PASSWORD", None)


    @property
    def endpoint(self) -> str:
        if not self._azure_search_endpoint:
            self._azure_search_endpoint = os.environ.get("ACS_ENDPOINT")

            if not self._azure_search_endpoint:
                raise ValueError("ACS_ENDPOINT is not defined")
        
        return self._azure_search_endpoint
    

    @property
    def key(self) -> Union[str, None]:
        if not self._azure_search_key:
            self._azure_search_key = os.environ.get("ACS_PASSWORD", None)
        return self._azure_search_key


config = AzureSearchConfig()
