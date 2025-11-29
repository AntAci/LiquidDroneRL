import sys
print(f"Python version: {sys.version}")
try:
    import numpy
    print("Numpy imported")
except ImportError as e:
    print(f"Numpy failed: {e}")

try:
    import gymnasium
    print("Gymnasium imported")
except ImportError as e:
    print(f"Gymnasium failed: {e}")

try:
    import PyFlyt
    print("PyFlyt imported")
except ImportError as e:
    print(f"PyFlyt failed: {e}")

try:
    import opensimplex
    print("Opensimplex imported")
except ImportError as e:
    print(f"Opensimplex failed: {e}")

try:
    import ncps
    print("Ncps imported")
except ImportError as e:
    print(f"Ncps failed: {e}")
