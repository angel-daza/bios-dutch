class BaseConfig(object): 
    'Base config class' 
    SECRET_KEY = 'A random secret key' 
    DEBUG = True 
    TESTING = False 
    NEW_CONFIG_VARIABLE = 'my value' 


class ProductionConfig(BaseConfig): 
    'Production specific config' 
    DEBUG = False 
    # SECRET_KEY = open('/path/to/secret_file.txt').read() 
 

class DevelopmentConfig(BaseConfig): 
    'Development environment specific config' 
    DEBUG = True 
    TESTING = True
    SECRET_KEY = 'Another random secret key'