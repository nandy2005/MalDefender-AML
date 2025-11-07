import os
import ember

data_dir = "./ember_files"

print("🔍 Checking EMBER dataset integrity...\n")

# ---------------------------
# Step 1: Check for required files
# ---------------------------
required_files = [
    "X_train.dat", "y_train.dat",
    "X_test.dat", "y_test.dat"
]

for f in required_files:
    path = os.path.join(data_dir, f)
    exists = os.path.exists(path)
    print(f"{f:15} exists: {exists}")
    if exists:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  └── Size: {size_mb:.2f} MB")

# ---------------------------
# Step 2: Attempt to read data
# ---------------------------
print("\n📂 Reading vectorized features (this may take a while)...\n")

try:
    # EMBER returns (X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)

    print("✅ Train and Test datasets loaded successfully.\n")
    print(f"📊 X_train shape: {X_train.shape}")
    print(f"📊 y_train shape: {y_train.shape}")
    print(f"📊 X_test  shape: {X_test.shape}")
    print(f"📊 y_test  shape: {y_test.shape}")

    print(f"\n🧾 First 10 train labels: {y_train[:10]}")
    print(f"🧾 First 10 test labels:  {y_test[:10]}")

except Exception as e:
    print("\n❌ Error reading vectorized features:")
    print(e)
