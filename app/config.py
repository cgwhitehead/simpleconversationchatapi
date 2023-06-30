from configparser import ConfigParser


def load_config(app):
    config = ConfigParser()
    config.read('C:/Users/chris/Workspace/SimpleConversationChatApi/app/static/config.ini')

    app.config['DEBUG'] = config.getboolean('DEFAULT', 'DEBUG')
    app.config['PATH_TO_MODEL'] = config.get('DEFAULT', 'PATH_TO_MODEL')
    app.config['PATH_TO_FILE_DB'] = config.get('DEFAULT', 'PATH_TO_FILE_DB')
