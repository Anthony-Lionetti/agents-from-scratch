
# Complete Logging Configuration Guide
import logging.config
import sys
from pathlib import Path

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

PRODUCTION_CONFIG = {
    'version': 1,  # Required - specifies config format version
    'disable_existing_loggers': True,  # Keep existing loggers active
    
    # FORMATTERS - Define how log messages look
    'formatters': {
        'minimal': {
            'format': '%(levelname)s: %(message)s'
        },
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    
    # FILTERS - Control which messages get processed
    'filters': {
        'require_debug_true': {
            '()': 'logging.Filter',
            'name': 'debug_filter'
        }
    },
    
    # HANDLERS - Define where logs go and how they're processed
    'handlers': {
        # Console output
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        
        # Console for errors only
        'console_error': {
            'class': 'logging.StreamHandler', 
            'level': 'ERROR',
            'formatter': 'detailed',
            'stream': 'ext://sys.stderr'
        },
        
        # Basic file logging
        'file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'logs/app.log',
            'mode': 'a',  # append mode
            'encoding': 'utf-8'
        },
        
        # Rotating file handler
        'rotating_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf-8'
        },
        
        # Time-based rotation
        'timed_rotating_file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'logs/app.log',
            'when': 'midnight',  # rotate at midnight
            'interval': 1,       # every 1 day
            'backupCount': 30,   # keep 30 days
            'encoding': 'utf-8'
        },
        
        # Separate error file
        'error_file': {
            'class': 'logging.FileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/errors.log',
            'mode': 'a',
            'encoding': 'utf-8'
        },
        
        # Email handler for critical errors
        'email': {
            'class': 'logging.handlers.SMTPHandler',
            'level': 'CRITICAL',
            'formatter': 'detailed',
            'mailhost': ('smtp.gmail.com', 587),
            'fromaddr': 'your-app@example.com',
            'toaddrs': ['admin@example.com'],
            'subject': 'Critical Error in Application',
            'credentials': ('username', 'password'),
            'secure': ()  # Use TLS
        },
        
        # Syslog handler (Unix/Linux/Mac)
        'syslog': {
            'class': 'logging.handlers.SysLogHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'address': '/dev/log',  # Mac/Linux path
            'facility': 'local0'
        },
        
        # Windows Event Log
        'eventlog': {
            'class': 'logging.handlers.NTEventLogHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'appname': 'MyPythonApp'
        },
        
        # HTTP handler (send logs to web service)
        'http': {
            'class': 'logging.handlers.HTTPHandler',
            'level': 'ERROR',
            'formatter': 'json',
            'host': 'logs.example.com',
            'url': '/api/logs',
            'method': 'POST'
        }
    },
    
    # LOGGERS - Configure specific loggers
    'loggers': {
        # Django-specific logger
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        
        # Database query logger
        'django.db.backends': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': False
        },
        
        # Your application logger
        'myapp': {
            'handlers': ['console', 'rotating_file', 'error_file'],
            'level': 'INFO',
            'propagate': False
        },
        
        # Third-party library (suppress verbose logging)
        'requests': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False
        },
        
        # Root logger (catches everything not specified above)
        '': {
            'handlers': ['console', 'rotating_file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# DEBUG CONFIGURATION
DEBUG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    
    'formatters': {
        'debug': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S.%f'  # Include microseconds
        },
        'console_debug': {
            'format': '%(levelname)s %(name)s: %(message)s (%(filename)s:%(lineno)d)'
        }
    },
    
    'handlers': {
        'console_debug': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'console_debug',
            'stream': 'ext://sys.stdout',
            
        },
        'debug_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'debug',
            'filename': 'logs/debug.log',
            'maxBytes': 5242880,  # 5MB (smaller for debug)
            'backupCount': 3,
            'encoding': 'utf-8'
        },
        'trace_file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'debug',
            'filename': 'logs/trace.log',
            'mode': 'w',  # overwrite mode for clean traces
            'encoding': 'utf-8'
        }
    },
    
    'loggers': {
        # Enable debug for your application
        'mcp-client': {
            'handlers': ['console_debug', 'debug_file', 'trace_file'],
            'level': 'DEBUG',
            'propagate': False
        },
        
        # Debug specific modules
        'mcp-cloent.database': {
            'handlers': ['debug_file'],
            'level': 'DEBUG',
            'propagate': True  # Also send to parent logger
        },
        
        # Root logger with debug enabled
        '': {
            'handlers': ['console_debug', 'debug_file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# USAGE EXAMPLES
def setup_production_logging():
    """Setup production logging configuration."""
    logging.config.dictConfig(PRODUCTION_CONFIG)
    logger = logging.getLogger('myapp')
    logger.info("Production logging configured")
    return logger

def setup_debug_logging():
    """Setup debug logging configuration."""
    logging.config.dictConfig(DEBUG_CONFIG)
    logger = logging.getLogger('mcp-client')
    logger.debug("Debug logging configured")
    return logger

# Environment-based configuration
def setup_logging(environment='prod'):
    """
    Setup logging based on environment.

    Args:
        envionment (str): Either 'prod' or 'dev'
    """
    if environment == 'dev':
        return setup_debug_logging()
    else:
        return setup_production_logging()