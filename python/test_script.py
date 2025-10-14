import pygame
import time
import numpy as np
import os
import glob
import pickle
import string
try:
    from tkinter import Tk, filedialog
except Exception:
    Tk = None
    filedialog = None
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot as plotly_offline_plot
except Exception:
    go = None
    px = None
    plotly_offline_plot = None

from interpolation import newton_interpolation, cubic_spline_interpolation

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
# Visualization helper
# ------------------------
def visualize_strokes(character):
    """
    Create a Plotly 3D visualization (browser) showing x (0-1), y (0-1) and time (s).
    Now takes a Character object instead of a list of strokes.
    """
    if not character or len(character) == 0:
        print("No strokes to visualize.")
        return

    if go is None or px is None or plotly_offline_plot is None:
        print("Plotly is not available. Install it with: pip install plotly")
        return

    traces = []
    palette = px.colors.qualitative.Plotly
    
    for i, stroke in enumerate(character):
        if len(stroke) == 0:
            continue
        traces.append(
            go.Scatter3d(
                x=stroke.x_coords,
                y=stroke.y_coords,
                z=stroke.timestamps,
                mode='lines+markers',
                line=dict(color=palette[i % len(palette)], width=4),
                marker=dict(size=3, color=palette[i % len(palette)]),
                name=f"stroke {i+1}"
            )
        )

    max_t = max((p.timestamp for p in character.all_points()), default=0.0)
    z_max = max_t * 1.05 if max_t > 0 else 1.0

    layout = go.Layout(
        title="Touch strokes: X, Y over time",
        scene=dict(
            xaxis=dict(title="X (normalized)", range=[0, 1]),
            yaxis=dict(title="Y (normalized)", range=[0, 1]),
            zaxis=dict(title="Time (s)", range=[0, z_max]),
        ),
        legend=dict(itemsizing='constant')
    )

    fig = go.Figure(data=traces, layout=layout)
    # This writes a temporary HTML file and opens it in the default browser.
    try:
        plotly_offline_plot(fig, auto_open=True, filename='handwriting_3d.html')
    except Exception as e:
        # As a fallback try the built-in renderer
        try:
            fig.show(renderer='browser')
        except Exception:
            print("Unable to open Plotly visualization:", e)


def visualize_parametric(character):
    """
    Create two Plotly 2D visualizations with both raw data and interpolated curves.
    Now takes a Character object instead of a list of strokes.
    """
    if not character or len(character) == 0:
        print("No strokes to visualize (parametric).")
        return

    if go is None or px is None or plotly_offline_plot is None:
        print("Plotly is not available. Install it with: pip install plotly")
        return

    palette = px.colors.qualitative.Plotly
    traces_x = []
    traces_y = []
    
    # Collect all points for interpolation
    all_points = list(character.all_points())
    all_times = np.array([p.timestamp for p in all_points])
    all_x = np.array([p.x_norm for p in all_points])
    all_y = np.array([p.y_norm for p in all_points])
    
    # Sort points by time for interpolation
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_x = all_x[sort_idx]
    all_y = all_y[sort_idx]
    
    # Generate smooth time points for plotting interpolated curves
    t_smooth = np.linspace(all_times.min(), all_times.max(), 500)
    
    # Create interpolation functions
    newton_x = newton_interpolation(all_times, all_x.copy())  # copy because newton modifies array
    newton_y = newton_interpolation(all_times, all_y.copy())
    spline_x = cubic_spline_interpolation(all_times, all_x)
    spline_y = cubic_spline_interpolation(all_times, all_y)
    
    # Add interpolated curves to traces
    traces_x.extend([
        go.Scatter(
            x=t_smooth, y=[newton_x(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(255,0,0,0.5)', width=2),
            name='Newton interpolation'
        ),
        go.Scatter(
            x=t_smooth, y=[spline_x(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(0,255,0,0.5)', width=2),
            name='Cubic spline'
        )
    ])
    
    traces_y.extend([
        go.Scatter(
            x=t_smooth, y=[newton_y(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(255,0,0,0.5)', width=2),
            name='Newton interpolation'
        ),
        go.Scatter(
            x=t_smooth, y=[spline_y(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(0,255,0,0.5)', width=2),
            name='Cubic spline'
        )
    ])
    
    # Add original data points
    for i, stroke in enumerate(character):
        if len(stroke) == 0:
            continue
        ts = [p.timestamp for p in stroke]
        xs = [p.x_norm for p in stroke]
        ys = [p.y_norm for p in stroke]
        color = palette[i % len(palette)]
        traces_x.append(
            go.Scatter(
                x=ts, y=xs,
                mode='markers',
                marker=dict(size=8, color=color),
                name=f"stroke {i+1} (data)"
            )
        )
        traces_y.append(
            go.Scatter(
                x=ts, y=ys,
                mode='markers',
                marker=dict(size=8, color=color),
                name=f"stroke {i+1} (data)"
            )
        )

    max_t = all_times.max()
    t_max = max_t * 1.05 if max_t > 0 else 1.0

    layout_x = go.Layout(
        title="X over Time (with interpolation)",
        xaxis=dict(title="Time (s)", range=[0, t_max]),
        yaxis=dict(title="X (normalized)", range=[-0.1, 1.1]),
    )
    layout_y = go.Layout(
        title="Y over Time (with interpolation)",
        xaxis=dict(title="Time (s)", range=[0, t_max]),
        yaxis=dict(title="Y (normalized)", range=[-0.1, 1.1]),
    )

    fig_x = go.Figure(data=traces_x, layout=layout_x)
    fig_y = go.Figure(data=traces_y, layout=layout_y)

    # Open each in its own html (browser) tab/window
    try:
        plotly_offline_plot(fig_x, auto_open=True, filename='handwriting_x_vs_t.html')
    except Exception:
        try:
            fig_x.show(renderer='browser')
        except Exception as e:
            print("Unable to open X(t) plot:", e)

    try:
        plotly_offline_plot(fig_y, auto_open=True, filename='handwriting_y_vs_t.html')
    except Exception:
        try:
            fig_y.show(renderer='browser')
        except Exception as e:
            print("Unable to open Y(t) plot:", e)

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

# ------------------------
# Main logic
# ------------------------

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# add near top-level globals (after imports / before main loop)
last_identified = []           # list of (char, confidence, sub_char)
identified_display_until = 0.0 # epoch time until which to display results

def reset_state():
    """Reset all state variables to initial values"""
    global current_character, current_stroke, recording, playback_start_time, recording_start_time
    global last_identified, identified_display_until
    current_character = Character()
    current_stroke = Stroke()
    recording = True
    playback_start_time = None
    recording_start_time = None
    # clear previous identification display
    last_identified = []
    identified_display_until = 0.0

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

def save_character(character, path=None):
    """Pickle the character to disk; chooses next name if path is None."""
    if path is None:
        path = next_character_filename()
    try:
        with open(path, 'wb') as f:
            pickle.dump(character, f)
        print(f"Saved character -> {path}")
        return path
    except Exception as e:
        print("Failed to save character:", e)
        return None

def load_character_via_dialog(initial_dir=None):
    """Open file dialog and return loaded Character or None."""
    if Tk is None or filedialog is None:
        print("tkinter file dialog unavailable. Cannot load file.")
        return None
    root = Tk()
    root.withdraw()
    try:
        fname = filedialog.askopenfilename(initialdir=(initial_dir or os.getcwd()),
                                           title="Select character .pkl file",
                                           filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
    finally:
        root.destroy()
    if not fname:
        return None
    try:
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        print(f"Loaded character <- {fname}")
        return obj
    except Exception as e:
        print("Failed to load character:", e)
        return None
    
def normalize_error(error):
    """Returns value scaled from 0 to 1 based on equation x/(x+1), 
    where original error is a positive real value.
    More sensitive to small errors ~0, saturates towards 1 for large errors."""
    return (9 + (error-0.9) / (error-0.9+1)) / 10

def character_error(character, reference_character):
    """
    Compare character to reference_character by computing mean squared error
    of x(t) and y(t) separately. Uses a cubic spline built from the reference
    (fallbacks to constant or linear interp) and evaluates it at the candidate
    character's timestamps.
    Returns (x_mse, y_mse).
    """
    # gather points
    char_pts = list(character.all_points())
    ref_pts = list(reference_character.all_points())

    if not char_pts or not ref_pts:
        return (float('inf'), float('inf'))

    char_times = np.asarray([p.timestamp for p in char_pts], dtype=float)
    char_x = np.asarray([p.x_norm for p in char_pts], dtype=float)
    char_y = np.asarray([p.y_norm for p in char_pts], dtype=float)

    ref_times = np.asarray([p.timestamp for p in ref_pts], dtype=float)
    ref_x = np.asarray([p.x_norm for p in ref_pts], dtype=float)
    ref_y = np.asarray([p.y_norm for p in ref_pts], dtype=float)

    # helper to build a callable interpolant for reference data
    def build_interpolant(t, v):
        if t.size == 0:
            return None
        if t.size == 1:
            const = float(v[0])
            return lambda tt: np.full_like(np.asarray(tt, dtype=float), const, dtype=float)
        # ensure t is strictly increasing for spline; if not, sort and unique
        sort_idx = np.argsort(t)
        t_s = t[sort_idx]
        v_s = v[sort_idx]
        # remove duplicate timestamps by keeping the last value
        unique_t, unique_idx = np.unique(t_s, return_index=True)
        if unique_t.size != t_s.size:
            # take first occurrence for each unique time
            t_s = t_s[unique_idx]
            v_s = v_s[unique_idx]
        try:
            return cubic_spline_interpolation(t_s, v_s)
        except Exception:
            # fallback to numpy interp (clamped extrapolation)
            return lambda tt: np.interp(np.asarray(tt, dtype=float), t_s, v_s, left=v_s[0], right=v_s[-1])

    sx = build_interpolant(ref_times, ref_x)
    sy = build_interpolant(ref_times, ref_y)
    if sx is None or sy is None:
        return (float('inf'), float('inf'))

    # evaluate interpolants at character timestamps
    try:
        pred_x = sx(char_times)
    except Exception:
        pred_x = np.asarray([sx(t) for t in char_times], dtype=float)
    try:
        pred_y = sy(char_times)
    except Exception:
        pred_y = np.asarray([sy(t) for t in char_times], dtype=float)

    # Mean squared error
    mse_x = float(np.mean((char_x - pred_x) ** 2))
    mse_y = float(np.mean((char_y - pred_y) ** 2))

    return (mse_x, mse_y)

def identify_character(character: Character):
    """
    Analyze the character and attempt to identify it.
    Returns a dictionary with key for each character and a confidence score (0-1).
    """
    d = {}
    for ascii_char, ref_char in REFERENCE_CHARACTERS.items():
        x_error, y_error = character_error(trim_leading_time(normalize_positions(character)), 
                                           trim_leading_time(normalize_positions(ref_char)))
        average_error = (x_error + y_error) / 2
        confidence = 1 - normalize_error(average_error)
        d[ascii_char] = confidence
    print("---- Character Results: ---\n")
    for k in sorted(d.keys()):
        print(f"  '{k}': {d[k]:.8f}")
    return d

def identify_screen_characters(screen_character: Character):
    """
    Capture the screen, analyze for characters, and attempt to identify them.
    Returns a list of Characters identified with high enough confidence.
    """
    def create_sub_character(screen_char, stroke_indices):
        sub_char = Character()
        for idx in stroke_indices:
            if 0 <= idx < len(screen_char.strokes):
                sub_char.add_stroke(screen_char.strokes[idx])
        all_pts = list(sub_char.all_points())
        min_x = min(p.x_norm for p in all_pts)
        max_x = max(p.x_norm for p in all_pts)
        min_y = min(p.y_norm for p in all_pts)
        max_y = max(p.y_norm for p in all_pts)
        sub_char.x_range = (min_x, max_x)
        sub_char.y_range = (min_y, max_y)
        return sub_char

    identified = []
    # Group up to three consecutive stroke combinations into one character for identification
    for i in range(0, len(screen_character.strokes)):
        sub_char = create_sub_character(screen_character, [i])
        result = identify_character(sub_char)
        if result:
            # For simplicity, assume result is a dict {char: confidence}
            best_match = max(result.items(), key=lambda kv: kv[1])
            if best_match[1] >= 0.7:  # confidence threshold
                identified.append((best_match[0], best_match[1], sub_char))
                print(f"Identified '{best_match[0]}' with confidence {best_match[1]:.2f}")

    for i in range(0, len(screen_character.strokes)-1):
        if i+1 >= len(screen_character.strokes):
            break
        sub_char = create_sub_character(screen_character, [i, i+1])
        result = identify_character(sub_char)
        if result:
            best_match = max(result.items(), key=lambda kv: kv[1])
            if best_match[1] >= 0.7:
                identified.append((best_match[0], best_match[1], sub_char))
                print(f"Identified '{best_match[0]}' with confidence {best_match[1]:.2f}")
    
    for i in range(0, len(screen_character.strokes)-2):
        if i+2 >= len(screen_character.strokes):
            break
        sub_char = create_sub_character(screen_character, [i, i+1, i+2])
        result = identify_character(sub_char)
        if result:
            best_match = max(result.items(), key=lambda kv: kv[1])
            if best_match[1] >= 0.7:
                identified.append((best_match[0], best_match[1], sub_char))
                print(f"Identified '{best_match[0]}' with confidence {best_match[1]:.2f}")

    return identified

reset_state()
running = True

while running:
    dt = clock.tick(60) / 1000.0
    now = time.time()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle record/playback
                recording = not recording
                if not recording:
                    playback_start_time = time.time()
                    print("Playback mode...")
                else:
                    print("Recording mode...")
            elif event.key == pygame.K_g:
                # Finalize any in-progress stroke
                if len(current_stroke) > 0:
                    current_character.add_stroke(current_stroke)
                    current_stroke = Stroke()
                
                print("Generating visualizations...")
                # Temporarily hide pygame window
                pygame.display.iconify()
                
                # Generate visualizations
                try:
                    visualize_strokes(current_character)
                    visualize_parametric(current_character)
                except Exception as e:
                    print(f"Visualization error: {e}")
                
                # Reset state and restore pygame window
                reset_state()
                pygame.display.set_mode((WIDTH, HEIGHT))
                continue
            elif event.key == pygame.K_r:
                # Reset everything
                print("Resetting...")
                reset_state()
                pygame.display.set_mode((WIDTH, HEIGHT))
                continue
            elif event.key == pygame.K_s:
                # Save current character to next available character_x.pkl
                # Ensure any in-progress stroke is included
                if len(current_stroke) > 0:
                    current_character.add_stroke(current_stroke)
                    current_stroke = Stroke()
                save_character(trim_leading_time(normalize_positions(current_character)))
                # continue without changing current recording/playback state
                continue
            elif event.key == pygame.K_l:
                # Load a saved Character via file dialog and start playback
                loaded = load_character_via_dialog()
                if loaded is None:
                    print("No character loaded.")
                    continue
                # Replace current_character and prepare for playback
                current_character = loaded
                current_stroke = Stroke()
                recording = False
                playback_start_time = time.time()
                recording_start_time = None
                print("Entering playback of loaded character...")
                continue
            elif event.key == pygame.K_i:
                # Finalize any in-progress stroke
                if len(current_stroke) > 0:
                    current_character.add_stroke(current_stroke)
                    current_stroke = Stroke()

                print("Identifying on-screen characters...")
                try:
                    results = identify_screen_characters(current_character)
                except Exception as e:
                    print("Identification failed:", e)
                    results = []

                # store and print results; display for 5 seconds
                last_identified = results
                identified_display_until = time.time() + 5.0
                print("Identification results:")
                for item in results:
                    # item expected as (char, confidence, sub_char)
                    try:
                        ch, conf, sub = item
                        print(f"  {ch}: {conf:.2f}")
                    except Exception:
                        print("  ", item)
                continue

    screen.fill((30, 30, 30))

    if recording:
        if recording_start_time is None:
            recording_start_time = now

        elapsed = now - recording_start_time

        # Display "Recording" text
        txt = font.render(f"Recording: {elapsed:.2f}s", True, (255, 255, 255))
        screen.blit(txt, (10, 10))

        pressed = pygame.mouse.get_pressed()[0]
        if pressed:
            mx, my = pygame.mouse.get_pos()
            x_norm = mx / WIDTH
            y_norm = my / HEIGHT
            tp = TouchPoint(x_norm, y_norm, now - recording_start_time)
            print(tp)
            current_stroke.add_point(tp)
        else:
            if len(current_stroke) > 0:
                current_character.add_stroke(current_stroke)
                current_stroke = Stroke()

        # Draw strokes live
        if len(current_stroke) > 1:
            pts = [(p.get_x(WIDTH), p.get_y(HEIGHT)) for p in current_stroke]
            pygame.draw.lines(screen, (0, 200, 0), False, pts, 2)
            
        for stroke in current_character:
            if len(stroke) > 1:
                pts = [(p.get_x(WIDTH), p.get_y(HEIGHT)) for p in stroke]
                pygame.draw.lines(screen, (0, 200, 0), False, pts, 2)

    else:
        if playback_start_time is None:
            playback_start_time = now

        elapsed = now - playback_start_time

        # Display elapsed time
        txt = font.render(f"Playback time: {elapsed:.2f}s", True, (255, 255, 255))
        screen.blit(txt, (10, 10))

        # Draw strokes progressively
        # for stroke in trim_leading_time(normalize_positions(current_character)):
        for stroke in current_character.denormalized().denormalized().denormalized():
        
            pts = [
                (p.get_x(WIDTH), p.get_y(HEIGHT))
                for p in stroke if p.timestamp <= elapsed
            ]
            if len(pts) > 1:
                pygame.draw.lines(screen, (200, 200, 0), False, pts, 2)

    # Overlay identification results when available
    if time.time() < identified_display_until and last_identified:
        # draw a translucent background box
        overlay_h = 20 + 18 * len(last_identified)
        overlay_w = 400
        overlay_x = 10
        overlay_y = HEIGHT - overlay_h - 10
        s = pygame.Surface((overlay_w, overlay_h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))  # semi-transparent
        screen.blit(s, (overlay_x, overlay_y))
        # header
        header_txt = font.render("Identification:", True, (255, 255, 255))
        screen.blit(header_txt, (overlay_x + 6, overlay_y + 4))
        # lines
        for i, item in enumerate(last_identified):
            try:
                ch, conf, _ = item
                line = f"{ch}: {conf:.2f}"
            except Exception:
                line = str(item)
            txt = font.render(line, True, (200, 200, 200))
            screen.blit(txt, (overlay_x + 6, overlay_y + 24 + i * 18))

    pygame.display.flip()

pygame.quit()
raise SystemExit
