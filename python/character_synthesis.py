import numpy as np

from data_structures import Character, Stroke, TouchPoint, load_reference_characters, load_characters
from interpolation import build_interpolant
from character_matching import adjust_time_delay, add_padding


USER_SPLINES = None
"""
Stores user splines, in format: 
    x_spline: callable
    y_spline: callable
    time_range: (start, end)
"""

def generate_spline(character_list: list[Character]):
    """Creates a spline based on collected data from user handwriting. 
    Input is list of Characters by same ASCII character.
    Returns an x and y parametric callable function, a time range array, and an x-to-y scaler proportion.
    """
    ascii_char = character_list[0].ascii_char
    if ascii_char == None:
        return None
    for character in character_list:
        if character.ascii_char != ascii_char:
            raise Exception("All characters in the spline generation list must be the same ASCII character.")
    
    # Adjust each character for time offsets
    if len(character_list) > 1:
        ref_char = character_list[0]
        pts = list(ref_char.all_points())
        ref_times = np.array([p.timestamp for p in pts])
        ref_x = np.array([p.x_norm for p in pts])
        ref_y = np.array([p.y_norm for p in pts])
        chars_adjusted = [] # Stores pairs of x&y parametrics aligned and ready for averaging
        adjusted_times_arrays = []
        min_x_time, min_y_time, max_x_time, max_y_time = ref_times[0], ref_times[0], ref_times[-1], ref_times[-1]

        # Create splines for x and y coordinates of reference character
        #chars_adjusted.append([build_interpolant(ref_times, ref_x), build_interpolant(ref_times, ref_y)])
        adjusted_times_arrays.append([ref_times, ref_x, np.array(ref_times), ref_y])
        x_y_props = [character_list[0].x_to_y_proportion()]

        # # Average values across all character data
        # for i, character in enumerate(character_list):
        #     if i == 0:
        #         continue  # skip reference character

        #     # Collect all points from the character
        #     points = list(character.all_points())
        #     if not points:
        #         continue
        #     # Extract timestamps and coordinates
        #     times = np.array([p.timestamp for p in points])
        #     char_x = np.array([p.x_norm for p in points])
        #     char_y = np.array([p.y_norm for p in points])
            
        #     # Adjust time delay
        #     times_x = adjust_time_delay(times, char_x, ref_times, ref_x)
        #     times_y = adjust_time_delay(times, char_y, ref_times, ref_y)
            
        #     times_x, char_x, ref_times, ref_x = add_padding(times_x, char_x, ref_times, ref_x)
        #     times_y, char_y, ref_times, ref_y = add_padding(times_y, char_y, ref_times, ref_y)
            
        #     # Create splines for x and y coordinates of current character
        #     chars_adjusted.append([build_interpolant(times_x, char_x), build_interpolant(times_y, char_y)])

        #     min_x_time = min(min_x_time, times_x[0])
        #     min_y_time = min(min_y_time, times_y[0])
        #     max_x_time = max(max_x_time, times_x[-1])
        #     max_y_time = max(max_y_time, times_y[-1])
        # Average values across all character data
        for i, character in enumerate(character_list):
            if i == 0:
                continue  # skip reference character

            # Collect all points from the character
            points = list(character.all_points())
            if not points:
                continue
            # Extract timestamps and coordinates
            times = np.array([p.timestamp for p in points])
            char_x = np.array([p.x_norm for p in points])
            char_y = np.array([p.y_norm for p in points])
            
            # Adjust time delay
            times_x = adjust_time_delay(times, char_x, ref_times, ref_x)
            times_y = adjust_time_delay(times, char_y, ref_times, ref_y)
            
            # times_x, char_x, ref_times, ref_x = add_padding(times_x, char_x, ref_times, ref_x)
            # times_y, char_y, ref_times, ref_y = add_padding(times_y, char_y, ref_times, ref_y)
            
            # # Create splines for x and y coordinates of current character
            # chars_adjusted.append([build_interpolant(times_x, char_x), build_interpolant(times_y, char_y)])

            adjusted_times_arrays.append([times_x, char_x, times_y, char_y])

            min_x_time = min(min_x_time, times_x[0])
            min_y_time = min(min_y_time, times_y[0])
            max_x_time = max(max_x_time, times_x[-1])
            max_y_time = max(max_y_time, times_y[-1])

            x_y_props.append(character.x_to_y_proportion())

        for char_arrays in adjusted_times_arrays:
            times_x, char_x, times_y, char_y = char_arrays

            # Extrapolate all time arrays to full time range
            while (times_x[0] > min_x_time + 0.001):
                times_x = np.insert(times_x, 0, times_x[0] - (1 / 60))
                char_x = np.insert(char_x, 0, char_x[0])
            while (times_x[-1] < max_x_time - 0.001):
                times_x = np.append(times_x, times_x[-1] + (1 / 60))
                char_x = np.append(char_x, char_x[-1])
            while (times_y[0] > min_y_time + 0.001):
                times_y = np.insert(times_y, 0, times_y[0] - (1 / 60))
                char_y = np.insert(char_y, 0, char_y[0])
            while (times_y[-1] < max_y_time - 0.001):
                times_y = np.append(times_y, times_y[-1] + (1 / 60))
                char_y = np.append(char_y, char_y[-1])
            
            # Rebuild splines with extrapolated data
            chars_adjusted.append([build_interpolant(times_x, char_x), build_interpolant(times_y, char_y)])

        # Interpolate all characters to the same time range
        num_points = int(60 * (max_x_time - min_x_time)) + 1  # assuming 60 Hz sampling rate
        x_times = np.linspace(min_x_time, max_x_time, num_points)
        y_times = np.linspace(min_y_time, max_y_time, num_points)

        chars_adjusted = np.array(chars_adjusted)
        adj_vals = np.array([[x_spline(x_times), y_spline(y_times)] for x_spline, y_spline in chars_adjusted])

        # Average across all chars
        avg_x = adj_vals[:, 0].mean(axis=0)
        avg_y = adj_vals[:, 1].mean(axis=0)
        x_to_y = sum(x_y_props) / len(x_y_props)
    else:
        points = list(character_list[0].all_points())
        x_times = np.array([p.timestamp for p in points])
        y_times = np.array([p.timestamp for p in points])
        avg_x = np.array([p.x_norm for p in points])
        avg_y = np.array([p.y_norm for p in points])
        x_to_y = character_list[0].x_to_y_proportion()

    x_times, avg_x, y_times, avg_y = add_padding(x_times, avg_x, y_times, avg_y)
    print(x_times[0], x_times[-1], y_times[0], y_times[-1], x_times.shape, avg_x.shape, y_times.shape, avg_y.shape)
    x_times = x_times - x_times[0]
    y_times = y_times - y_times[0]
    assert x_times.shape == y_times.shape
    assert abs(x_times[0] - y_times[0]) < 0.017 * 2
    assert abs(x_times[-1] - y_times[-1]) < 0.017 * 2

    return build_interpolant(x_times, avg_x), build_interpolant(y_times, avg_y), \
           (0, x_times[-1]), x_to_y

def load_user_data(user: str):
    """Loads user handwriting data from storage.
    If spline data has not been created yet, generate and save pickled splines.
    Returns a dictionary of callable splines.
    """
    if user == "reference":
        # Load reference user data from character_references/ directory
        characters_dict = load_reference_characters()
    else:
        # loading from user_data/ directory
        characters_dict = load_characters(user=user)
    
    splines = {}
    for char, character_list in characters_dict.items():
        if not character_list:
            continue
        if not isinstance(character_list, list): # If coming from reference loading
            character_list = [character_list]
        if not character_list:
            continue
        x_spline, y_spline, time_range, x_to_y = generate_spline(character_list)
        
        if x_spline is not None and y_spline is not None:
            splines[char] = (x_spline, y_spline, time_range, x_to_y)

    return splines

def get_user_splines(user: str="reference"):
    """Returns the user splines, loading them if necessary.
    """
    global USER_SPLINES
    if USER_SPLINES is None:
        USER_SPLINES = load_user_data(user)
    return USER_SPLINES

def reset_loaded_user_splines():
    """Resets the loaded user splines, forcing a reload on next access.
    """
    global USER_SPLINES
    USER_SPLINES = None
    
def get_pixel_points(character: Character, rect_left: float, rect_top: float, rect_width: float, rect_height: float) -> list[dict]:
    """Convert a Character object into a list of pixel-space points for drawing.
    
    Args:
        character: The Character object to convert
        rect_left: Left coordinate of drawing rectangle
        rect_top: Top coordinate of drawing rectangle  
        rect_width: Width of drawing rectangle
        rect_height: Height of drawing rectangle

    Returns:
        List of dicts containing {x: pixel_x, y: pixel_y, timestamp: t} for each point
    """
    pixel_points = []
    all_pts = np.array([[p.x_norm, p.y_norm] for p in character.all_points()])
    
    # Iterate through strokes to maintain stroke count information
    for stroke_idx, stroke in enumerate(character.strokes):
        # Convert points in this stroke
        stroke_pts = np.array([[p.x_norm, p.y_norm] for p in stroke])
        
        if len(stroke_pts) == 0:
            continue
            
        stroke_pts = stroke_pts / all_pts.max() if len(all_pts) > 0 else stroke_pts  # Normalize to [0, 1] range

        # Scale normalized points to fit in drawing rectangle
        x_vals = rect_left + stroke_pts[:, 0] * rect_width 
        y_vals = rect_top + stroke_pts[:, 1] * rect_height

        # Get timestamps from original points
        timestamps = [p.timestamp for p in stroke]

        # Create point dicts with pixel coordinates and stroke count
        for x, y, t in zip(x_vals, y_vals, timestamps):
            pixel_points.append({
                'x': float(x),
                'y': float(y),
                'timestamp': float(t),
                'stroke_count': stroke_idx
            })

    return pixel_points

def generate_sentence_character(text: str, user: str="reference") -> list[Character]:
    """Generate Character object that comprises many Characters
      from the input text using user splines, ready for drawing on screen.
    
    Args:
        text: The input text to convert to characters
        
    Returns:
        A list of Character objects representing each character in the input text
    """
    splines = get_user_splines(user)
    character_sentence = Character()
    start_offset = 0.0 # seconds
    horz_offset = 0.0 # horizontal offset for multiple characters, unitless
    time_between_characters = 0.2  # seconds
    
    for char in text:
        if char in splines:
            x_spline, y_spline, time_range, x_to_y = splines[char]
            writing_time = np.linspace(time_range[0], time_range[1], int(60 * (time_range[1] - time_range[0])) + 1)
            xs = x_spline(writing_time) * x_to_y
            ys = y_spline(writing_time)

            s = Stroke()
            for x, y, time in zip(xs, ys, writing_time):
                s.add_point(TouchPoint(x+horz_offset, y, time + start_offset))
            character_sentence.add_stroke(s)
            start_offset += time_range[1] - time_range[0] + time_between_characters
            horz_offset = horz_offset + xs.max()  # Increment horizontal offset for next character

    return character_sentence
