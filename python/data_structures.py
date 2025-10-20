import os 
import glob
import pickle
import string
import numpy as np

# ------------------------
# Data classes
# ------------------------

class TouchPoint:
    def __init__(self, x_norm, y_norm, timestamp):
        self.x_norm = x_norm  # 0–1
        self.y_norm = y_norm  # 0–1
        self.timestamp = timestamp

    def get_x(self, width):
        return int(self.x_norm * width)

    def get_y(self, height):
        return int(self.y_norm * height)

    def __str__(self):
        return f"({self.x_norm:.2f}, {self.y_norm:.2f}) @ {self.timestamp:.2f}s"


class Stroke:
    """Contains an ordered sequence of TouchPoints making up a single stroke"""
    def __init__(self):
        self.points = []
    
    def add_point(self, point):
        self.points.append(point)
    
    def __len__(self):
        return len(self.points)
    
    def __iter__(self):
        return iter(self.points)
    
    @property
    def timestamps(self):
        return [p.timestamp for p in self.points]
    
    @property
    def x_coords(self):
        return [p.x_norm for p in self.points]
    
    @property
    def y_coords(self):
        return [p.y_norm for p in self.points]


class Character:
    """Contains multiple strokes that make up a single character"""
    def __init__(self):
        self.strokes = []
        # store original ranges used for normalization: (min, max) or None
        self.x_range = None
        self.y_range = None
        self.ascii_char = None  # optional: store which character this represents

    def add_stroke(self, stroke):
        if len(stroke) > 0:  # Only add non-empty strokes
            self.strokes.append(stroke)
    
    def __len__(self):
        return len(self.strokes)
    
    def __iter__(self):
        return iter(self.strokes)

    def __str__(self):
        return f"Character with {len(self.strokes)} strokes\n" + \
                f"{self.x_range}, {self.y_range}"
    
    def __repr__(self):
        return f"Character({self.__hash__()})"
    
    def __hash__(self):
        self.all_points()
        l = [p.x_norm * p.y_norm / max(p.timestamp, 0.3) for p in self.all_points()]
        return int(float(np.mean(l))*2e32)
    
    def all_points(self):
        """Iterator over all points in all strokes"""
        for stroke in self.strokes:
            yield from stroke
    
    def x_to_y_proportion(self):
        """Return the proportion of the x range to the y range."""
        if self.x_range is None or self.y_range is None:
            return 1
        return (self.x_range[1] - self.x_range[0]) / (self.y_range[1] - self.y_range[0])
    
    def get_bounding_box(self):
        """Return (min_x, min_y, max_x, max_y) across all points, or None if no points."""
        all_pts = list(self.all_points())
        if not all_pts:
            return None
        min_x = min(p.x_norm for p in all_pts)
        max_x = max(p.x_norm for p in all_pts)
        min_y = min(p.y_norm for p in all_pts)
        max_y = max(p.y_norm for p in all_pts)
        return (min_x, min_y, max_x, max_y)

    def denormalized(self):
        """
        Return a new Character with x_norm/y_norm values mapped back to the original
        scale using self.x_range and self.y_range.
        - If x_range or y_range is None, that axis is left unchanged.
        - If range min==max, the axis values are set to that constant.
        The returned Character will have the same x_range/y_range copied.
        """
        if self.x_range is None and self.y_range is None:
            # nothing to do
            return self

        new_char = Character()
        new_char.x_range = None
        new_char.y_range = None

        for stroke in self.strokes:
            new_stroke = Stroke()
            for p in stroke:
                # X denormalize
                if self.x_range is None:
                    x = p.x_norm
                else:
                    min_x, max_x = self.x_range
                    rx = max_x - min_x
                    if rx == 0:
                        x = min_x
                    else:
                        x = p.x_norm * rx + min_x
                # Y denormalize
                if self.y_range is None:
                    y = p.y_norm
                else:
                    min_y, max_y = self.y_range
                    ry = max_y - min_y
                    if ry == 0:
                        y = min_y
                    else:
                        y = p.y_norm * ry + min_y

                new_p = TouchPoint(x, y, p.timestamp)
                new_stroke.add_point(new_p)
            new_char.add_stroke(new_stroke)
        return new_char
    

def trim_leading_time(character):
    """
    Return a new Character with all timestamps shifted so the first timestamp becomes zero.
    Does not modify the input Character.
    """
    # collect all points to find minimal timestamp
    all_pts = list(character.all_points())
    if not all_pts:
        # empty character -> return shallow copy
        new_char = Character()
        new_char.x_range = character.x_range
        new_char.y_range = character.y_range
        return new_char

    min_t = min(p.timestamp for p in all_pts)

    new_char = Character()
    # preserve stored ranges if present
    new_char.x_range = character.x_range
    new_char.y_range = character.y_range

    for stroke in character:
        new_stroke = Stroke()
        for p in stroke:
            new_p = TouchPoint(p.x_norm, p.y_norm, p.timestamp - min_t)
            new_stroke.add_point(new_p)
        new_char.add_stroke(new_stroke)
    return new_char

def normalize_positions(character):
    """
    Return a new Character with x_norm and y_norm scaled to [0,1] based on the min/max
    found across all points in the input character. The returned Character.x_range and
    .y_range are set to (min, max) for each axis so the normalization can be reversed:
    original = normalized * (max - min) + min
    """
    all_pts = list(character.all_points())
    if not all_pts:
        new_char = Character()
        new_char.x_range = character.x_range
        new_char.y_range = character.y_range
        return new_char

    min_x = min(p.x_norm for p in all_pts)
    max_x = max(p.x_norm for p in all_pts)
    min_y = min(p.y_norm for p in all_pts)
    max_y = max(p.y_norm for p in all_pts)

    # avoid division by zero: if range is zero, map all to 0.5
    rx = max_x - min_x
    ry = max_y - min_y

    new_char = Character()
    new_char.x_range = (min_x, max_x)
    new_char.y_range = (min_y, max_y)

    for stroke in character:
        new_stroke = Stroke()
        for p in stroke:
            if rx == 0:
                nx = 0.5
            else:
                nx = (p.x_norm - min_x) / rx
            if ry == 0:
                ny = 0.5
            else:
                ny = (p.y_norm - min_y) / ry
            new_p = TouchPoint(nx, ny, p.timestamp)
            new_stroke.add_point(new_p)
        new_char.add_stroke(new_stroke)
    return new_char

# ------------------------
# Load reference characters
# ------------------------
def load_reference_characters(ref_dir=None):
    """
    Scan ref_dir for character_*.pkl files and unpickle them into a dict
    mapping real characters ('a'..'z','A'..'Z') -> Character objects.
    Supports filenames:
      - character_a.pkl     -> 'a'
      - character_A.pkl     -> 'A'
      - character_ac.pkl    -> 'A'   (legacy 'c' suffix mapped to uppercase)
    """
    if ref_dir is None:
        # directory relative to this script
        ref_dir = os.path.join(os.path.dirname(__file__), "character_references")
    ref_dir = os.path.abspath(ref_dir)
    d = {}
    if not os.path.isdir(ref_dir):
        print(f"Reference directory not found: {ref_dir}")
        return d

    pattern = os.path.join(ref_dir, "character_*.pkl*")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No reference files found in {ref_dir}")
        return d

    for path in files:
        name = os.path.basename(path)
        # remove possible trailing .tmp_rename
        if name.endswith(".tmp_rename"):
            name = name[:-len(".tmp_rename")]
        base, ext = os.path.splitext(name)  # base like character_a or character_ac
        if not base.startswith("character_"):
            continue
        token = base[len("character_"):]
        key = None
        # direct single-letter token (lower or upper)
        if len(token) == 1 and token.isalpha():
            key = token
        # legacy: trailing 'c' indicates capital (e.g. 'ac' -> 'A')
        elif len(token) == 2 and token[1].lower() == 'c' and token[0].isalpha():
            key = token[0].upper()
        # as a fallback, if token matches a single letter in ascii_letters, pick it
        elif token in string.ascii_letters:
            key = token
        else:
            print(f"Skipping unrecognized reference filename: {os.path.basename(path)}")
            continue

        # try loading
        
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
                print(obj.__hash__())
            # basic sanity: has strokes attribute
            if not hasattr(obj, "strokes"):
                print(f"File {os.path.basename(path)} did not contain a Character-like object, skipping.")
                continue
            # map key: prefer exact case for keys (lowercase 'a'..'z' and uppercase 'A'..'Z')
            # ensure key is single-letter exact case
            if len(key) == 1:
                d[key] = obj
                print(f"Loaded reference for '{key}' from {os.path.basename(path)}")
        except Exception as e:
            print(f"Failed to load reference {os.path.basename(path)}: {e}")
    return d

# Populate global reference mapping (available to rest of script)
REFERENCE_CHARACTERS = load_reference_characters()
