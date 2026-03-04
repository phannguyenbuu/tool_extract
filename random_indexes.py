import json
import random

# Tạo mapping 0->108 với random unique values 0-108
numbers = list(range(109))  # 0 to 108 inclusive
random.shuffle(numbers)
mapping = {i: numbers[i] for i in range(109)}

# Lưu thành JSON
with open('random_mapping_0_108.json', 'w') as f:
    json.dump(mapping, f, indent=2)

print("✅ Đã tạo file 'random_mapping_0_108.json'")
print("Ví dụ mapping:")
for i in range(0, 109, 10):
    print(f"  {i}: {mapping[i]}")
