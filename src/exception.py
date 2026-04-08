import sys

def error_message_details(error,error_details:sys):
    exc_tb = error.__traceback__
    while exc_tb.tb_next:
        exc_tb = exc_tb.tb_next
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = f"Exception Occured in File {file_name}\n\t\t At line no -> {line_no}\nMessage: {error}"

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details=error_details)
    
    def __str__(self):
        return self.error_message
        