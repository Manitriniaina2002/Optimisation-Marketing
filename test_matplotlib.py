import pandas as pd
import matplotlib

print("Testing background_gradient...")

# Test basic functionality
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

try:
    styled = df.style.background_gradient(cmap='YlGnBu')
    print("✅ background_gradient works!")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Pandas version: {pd.__version__}")
