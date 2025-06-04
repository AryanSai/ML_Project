from src.exception import CustomException
import dill

def save_object(file_path, obj):
    try:
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except ImportError:
        print("os module is not available. Cannot create directories.")
        return
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)