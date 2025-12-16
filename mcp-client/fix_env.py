
import os

def check_file(path):
    print(f"Checking {path}")
    content = None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("Successfully read as UTF-8")
        return content
    except UnicodeDecodeError:
        print("Failed to read as UTF-8")

    try:
        with open(path, 'r', encoding='utf-16') as f:
            content = f.read()
        print("Successfully read as UTF-16")
        return content
    except Exception as e:
        print(f"Failed to read as UTF-16: {e}")
        
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        print(f"Raw bytes: {raw[:20]}...")
        # try decoding with 'latin-1' just to recover something
        return raw.decode('latin-1')
    except Exception as e:
        print(f"Failed to read raw: {e}")
        return None

content = check_file('.env')
if content:
    print("Content recovered. Writing back as UTF-8...")
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed .env encoding.")
else:
    print("Could not recover content.")
