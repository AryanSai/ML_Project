#provides variables and functions that interact with the Python runtime environment
import sys 
from logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,error_detail = error_detail.exc_info()
    error_message = f"Error occurred in script: {error_detail.tb_frame.f_code.co_filename} at line {error_detail.tb_lineno}: {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    
# if __name__ == "__main__":
#     try:
#         raise ValueError("An example error")
#     except Exception as e:
#         logging.info("An error occurred: %s", e)
#         raise CustomException(e, sys) from e