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