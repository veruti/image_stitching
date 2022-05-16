from pydantic import BaseConfig


class Settings(BaseConfig):
    ALLOWED_IMAGES_TYPE = "png"


settings = Settings()
