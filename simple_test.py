import sys
import os

def main():
    print("=== Simple Python Test ===")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Create a test file
    test_file = "test_output.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Test successful!")
        print(f"Successfully wrote to {test_file}")
        
        # Read the file back
        with open(test_file, 'r') as f:
            content = f.read()
        print(f"File content: {content}")
        
        # Delete the test file
        os.remove(test_file)
        print(f"Successfully deleted {test_file}")
        
        # List directory contents
        print("\nDirectory contents:")
        for item in os.listdir('.'):
            print(f"- {item}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
