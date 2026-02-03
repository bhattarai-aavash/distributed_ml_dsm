import pandas as pd

# Test timestamp parsing
test_timestamp = "11 Sep 2025 15:14:49"
print(f"Original timestamp: {test_timestamp}")

try:
    parsed = pd.to_datetime(test_timestamp, format="%d %b %Y %H:%M:%S")
    print(f"Parsed timestamp: {parsed}")
    print(f"Type: {type(parsed)}")
    
    # Test elapsed time calculation
    timestamps = [test_timestamp, "11 Sep 2025 15:15:00", "11 Sep 2025 15:15:30"]
    df = pd.DataFrame({"timestamp": timestamps})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d %b %Y %H:%M:%S")
    df['elapsed_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    print(f"\nDataFrame:")
    print(df)
    print(f"\nElapsed times: {df['elapsed_sec'].tolist()}")
    
except Exception as e:
    print(f"Error: {e}")
