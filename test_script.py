import os

def main():
    # Get current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # List files in the current directory
    print("\nFiles in the current directory:")
    for item in os.listdir('.'):
        print(f"- {item}")
    
    # Try to create a test file
    test_file = os.path.join(cwd, 'test_output.txt')
    try:
        with open(test_file, 'w') as f:
            f.write("This is a test file.\n")
        print(f"\nSuccessfully created file: {test_file}")
    except Exception as e:
        print(f"\nError creating file: {str(e)}")

if __name__ == "__main__":
    main()
