import sys

def main():
    # Print basic information
    print("=== MINIMAL PYTHON TEST ===")
    print(f"Python version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Command line arguments: {sys.argv}")
    
    # Test basic Python operations
    try:
        # Test basic math
        result = 2 + 2
        print(f"2 + 2 = {result}")
        
        # Test list comprehension
        squares = [x**2 for x in range(5)]
        print(f"Squares: {squares}")
        
        # Test file operations
        test_content = "This is a test file."
        with open('test_output.txt', 'w') as f:
            f.write(test_content)
        print("Successfully wrote to test_output.txt")
        
        # Read back the file
        with open('test_output.txt', 'r') as f:
            content = f.read()
        print(f"Read back: {content}")
        
        # Clean up
        import os
        os.remove('test_output.txt')
        print("Cleaned up test file")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()
