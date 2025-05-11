import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print("sys.path =", sys.path)

try:
    from src.exception import CustomException
    print("Import worked ✅")
except ModuleNotFoundError as e:
    print("Import failed ❌", e)
