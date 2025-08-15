import sys
import os
import pkg_resources

def main():
    print("=== Python Environment Test ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print("\n=== Installed Packages ===")
    for pkg in sorted(pkg_resources.working_set, key=lambda x: x.key):
        print(f"{pkg.key}=={pkg.version}")
    
    # Test file writing
    print("\n=== File System Test ===")
    test_file = "test_output.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Test successful!")
        print(f"Successfully wrote to {test_file}")
        os.remove(test_file)
        print(f"Successfully deleted {test_file}")
    except Exception as e:
        print(f"Error writing/deleting test file: {e}")

if __name__ == "__main__":
    main()
