from openai import OpenAI
from depths.logger.core import DepthsLogger
from typing import Optional

class LoggedOpenAI:
    '''
    Logged OpenAI client
    '''
    def __init__(self,*args,logger: Optional[DepthsLogger] = None, **kwargs):
        self.client=OpenAI(*args, **kwargs)
        self.logger=logger
    
    def __getattr__(self, name: str):
        attr=getattr(self.client, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return result
            return wrapper
        return attr