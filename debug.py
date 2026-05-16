import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

TEST_DATASET = "test_samples"
CLASSES = ["cow", "goat", "elephant", "wildboar"]

for cls in CLASSES:
    folder = os.path.join(TEST_DATASET, cls)
    print(f"\n📂 {cls}:")
    if not os.path.exists(folder):
        print(" ❌ Folder missing")
        continue
    
    for sub in ["audio", "images"]:
        subfolder = os.path.join(folder, sub)
        if os.path.exists(subfolder):
            files = os.listdir(subfolder)
            print(f"  ✔ {sub} ({len(files)} files)")
        else:
            if sub == "images" and cls in ["goat", "wildboar"]:
                pass # expected
            else:
                print(f"  ❌ {sub} missing")
