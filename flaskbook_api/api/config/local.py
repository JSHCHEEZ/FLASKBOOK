from flaskbook_api.api.config.base import Config


class Local(Config):
    TESTING = True
    DEBUG = True
