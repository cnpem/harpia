import harpia
import pkgutil
import os

print('\nCorrectly imported Harpia module!')

print("\n📦 Harpia module information")
print("---------------------------")
print(f"Module file: {harpia.__file__}")
print(f"Package directory: {os.path.dirname(harpia.__file__)}")

print("\n📚 Submodules found:")
package_path = os.path.dirname(harpia.__file__)
for module in pkgutil.iter_modules([package_path]):
    print(f"- {module.name}")

print("\n📂 Top-level attributes and functions:")
print(dir(harpia))

print("\n📂 filtersOperations:")
print(dir(harpia.filters.filtersOperations))

print("\n📂 filtersChunked:")
print(dir(harpia.filters.filtersChunked))

print("\n📂 functions operations_binary:")
print(dir(harpia.morphology.operations_binary))

print("\n📂 operations_grayscale:")
print(dir(harpia.morphology.operations_grayscale))

