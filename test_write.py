import os

def main():
    # Test file writing
    test_file = 'test_output.txt'
    try:
        with open(test_file, 'w') as f:
            f.write("This is a test file.\n")
            f.write(f"Current directory: {os.getcwd()}\n")
            f.write("Files in directory:\n")
            for item in os.listdir('.'):
                f.write(f"- {item}\n")
        print(f"Successfully wrote to {test_file}")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        import sys
        sys.exit(1)
