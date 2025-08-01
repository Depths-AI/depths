from depths.logger.core import DepthsLogger

from openai import OpenAI

from typing import Optional, Callable, Dict, Tuple, Any
import functools

def recursive_getattr(obj, attr, *args):
    '''
    Util function to recursively get attributes from an object.
    '''
    def _get(o, a): return getattr(o, a, *args)
    return functools.reduce(_get, [obj] + attr.split('.'))

def recursive_setattr(obj,attr,value):
    '''
    Util function to recursively set attributes on an object.
    '''
    pre, _, post = attr.rpartition('.')
    target = recursive_getattr(obj, pre) if pre else obj
    setattr(target, post, value)

def make_wrapper(original, handler, path):
    '''
    Wrapper factory to define custom logging methods for specific
    LLM client methods.

    Agnostic of LLM client.
    '''
    def wrapped(*args, **kwargs):
        result=original(*args, **kwargs)
        handler(path, args, kwargs, result)
        return result
    return wrapped


class LoggedOpenAI:
    '''
    Logged OpenAI client
    '''
    def openai_handle_chat_create(
            self,
            path:str,
            args: Tuple[Any],
            kwargs: Dict[str, Any],
            result: Any
            )->None:
            '''
            Custom logging handler for `OpenAI()` `chat.completions.create` method
            '''
            result_dict=result.to_dict()

            if self.logger.llm_logging_config.store_input_text==False:
                kwargs.pop("messages")
            
            if self.logger.llm_logging_config.store_output_text==False:
                result_dict.pop("choices")
            
            print(kwargs)
            print(result_dict)
        
    def __getattr__(self, name: str):
        attr=getattr(self.client, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return result
            return wrapper
        return attr

    def __init__(self,*args,logger: Optional[DepthsLogger] = None, **kwargs):
        self.client=OpenAI(*args, **kwargs)
        if logger is None:
            logger=DepthsLogger()
        self.logger=logger

        self.OPENAI_METHOD_REGISTRY: Dict[str, Callable] = {
            "chat.completions.create": self.openai_handle_chat_create,
        }

        for path, handler in self.OPENAI_METHOD_REGISTRY.items():
            original_func=recursive_getattr(self.client, path)
            wrapped_func=make_wrapper(original_func, handler, path)
            recursive_setattr(self.client, path, wrapped_func)
    
        
