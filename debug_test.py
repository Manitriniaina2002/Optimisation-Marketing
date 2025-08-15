import sys
import os

def main():
    # Print to stdout
    print("=== DEBUG TEST START ===")
    print(f"Python version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test file writing
    try:
        with open('debug_output.txt', 'w') as f:
            f.write("Debug test successful!")
        print("Successfully wrote to debug_output.txt")
    except Exception as e:
        print(f"Error writing file: {e}")
    
    # Test output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(os.path.join(output_dir, 'debug_output.txt'), 'w') as f:
            f.write("Debug test in output directory successful!")
        print("Successfully wrote to output/debug_output.txt")
    except Exception as e:
        print(f"Error writing to output directory: {e}")
    
    print("=== DEBUG TEST COMPLETE ===")

if __name__ == "__main__":
    main()
