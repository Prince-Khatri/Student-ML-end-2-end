import functools
from .exception import CustomException
import sys

def handle_exception(func):
    @functools.wraps(func)
    def new_func(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            raise CustomException(e,sys)
    return new_func

    
        
        