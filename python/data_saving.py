import os
import glob
import pickle


def save_character(character, user="default", path=None, char_letter=None):
    """Pickle the character to disk; chooses next name if path is None."""
    if user != "default":
        user_dir = os.path.join(os.getcwd(), "user_data", user)
        os.makedirs(user_dir, exist_ok=True)
        if char_letter is None:
            char_letter = input("Enter character you are saving: ")
        character.ascii_char = char_letter
        if char_letter.isupper():
            char_letter = char_letter.lower() + 'c'
            pattern = os.path.join(user_dir, f"character_{char_letter}*.pkl")
        else:
            pattern = os.path.join(user_dir, f"character_{char_letter}[0-9]*.pkl")
        
        existing = glob.glob(pattern)
        nums = []
        for p in existing:
            name = os.path.basename(p)
            try:
                n = int(name.split('_')[1].split('.')[0][-1])
                nums.append(n)
            except Exception:
                continue
        i = 0
        while i in nums:
            i += 1
        path = os.path.join(user_dir, f"character_{char_letter}{i}.pkl")
    elif path is None:
        path = next_character_filename()
    try:
        with open(path, 'wb') as f:
            pickle.dump(character, f)
        print(f"Saved character -> {path}")
        return path
    except Exception as e:
        print("Failed to save character:", e)
        return None
    
def next_character_filename(base_dir=None):
    """Return next available filename like character_0.pkl, character_1.pkl, ..."""
    if base_dir is None:
        base_dir = os.getcwd()
    pattern = os.path.join(base_dir, "character_*.pkl")
    existing = glob.glob(pattern)
    nums = []
    for p in existing:
        name = os.path.basename(p)
        try:
            n = int(name.split('_')[1].split('.')[0])
            nums.append(n)
        except Exception:
            continue
    i = 0
    while i in nums:
        i += 1
    return os.path.join(base_dir, f"character_{i}.pkl")
