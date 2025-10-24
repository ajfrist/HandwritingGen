import numpy as np

from data_structures import Character, normalize_positions, trim_leading_time, load_reference_characters, REFERENCE_CHARACTERS
from interpolation import cubic_spline_interpolation

# Global cache for adjusted splines, temparary solution to share up-to-date data with visualization 
all_character_adjusted_splines = {}

def get_current_adjusted_splines():
    return all_character_adjusted_splines

def get_average_time_step(times):
    """Estimate average time step from a time array."""
    times = np.asarray(times, dtype=float)
    diffs = np.diff(np.sort(times))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.01
    return float(np.median(diffs))

def adjust_time_delay(times, array, ref_times, ref_array):
    """
    Align `array` to `ref_array` by finding the time delay and time scaling that maximizes their
    cross-correlation. Returns a new times array shifted and scaled for optimal alignment.

    Parameters:
    - times: 1D array of timestamps for `array`
    - array: 1D array of sample values (may contain None or np.nan for gaps)
    - ref_array: 1D array of reference sample values (may contain None/np.nan)
    - ref_times: optional 1D array of timestamps for ref_array; if None an equispaced
      axis will be created for ref_array spanning the same interval as `times`.

    Returns:
    - times_shifted: numpy array same shape as `times`, shifted and scaled to best align with ref_array.
      The time values are scaled by a factor between 0.5 and 2.0 to maximize correlation.
    """
    # Basic validation
    if times is None or array is None or ref_array is None:
        return times
    times = np.asarray(times, dtype=float)
    a = np.asarray([np.nan if v is None else v for v in array], dtype=float)
    r = np.asarray([np.nan if v is None else v for v in ref_array], dtype=float)

    # Determine ref_times
    if ref_times is None:
        # If lengths match, assume same timebase
        if r.size == times.size and np.nanmedian(np.diff(times)) > 0:
            ref_times = times.copy()
        else:
            # fallback: spread ref times over same span as times
            if times.size > 1:
                t_start, t_end = times.min(), times.max()
            else:
                t_start, t_end = 0.0, float(r.size - 1)
            if r.size > 1:
                ref_times = np.linspace(t_start, t_end, r.size)
            else:
                ref_times = np.array([t_start], dtype=float)
    else:
        ref_times = np.asarray(ref_times, dtype=float)

    # Masks for finite values
    mask_a = np.isfinite(a) & np.isfinite(times)
    mask_r = np.isfinite(r) & np.isfinite(ref_times)

    if mask_a.sum() < 1 or mask_r.sum() < 1:
        # Not enough data to correlate
        return times

    # Use valid pairs only
    t_a = times[mask_a]
    v_a = a[mask_a]
    t_r = ref_times[mask_r]
    v_r = r[mask_r]

    # Build a common uniform grid
    # Estimate dt as median of positive diffs from both time arrays
    diffs = []
    if t_a.size > 1:
        diffs.append(np.median(np.diff(np.sort(t_a))))
    if t_r.size > 1:
        diffs.append(np.median(np.diff(np.sort(t_r))))
    if diffs:
        dt = float(np.median(diffs))
        if dt <= 0 or not np.isfinite(dt):
            dt = 0.01
    else:
        dt = 0.01

    t0 = min(t_a.min(), t_r.min())
    t1 = max(t_a.max(), t_r.max())
    if t1 <= t0:
        # degenerate span
        return times
    grid = np.arange(t0, t1 + dt/2.0, dt)

    # Interpolate onto grid
    def interp_to_grid(t_src, v_src):
        if t_src.size == 1:
            return np.full_like(grid, float(v_src[0]), dtype=float)
        # ensure increasing
        order = np.argsort(t_src)
        t_sorted = t_src[order]
        v_sorted = v_src[order]
        # unique times
        unique_t, unique_idx = np.unique(t_sorted, return_index=True)
        t_u = unique_t
        v_u = v_sorted[unique_idx]
        # interpolation with clamped ends
        return np.interp(grid, t_u, v_u, left=v_u[0], right=v_u[-1])

    grid_a = interp_to_grid(t_a, v_a)
    grid_r = interp_to_grid(t_r, v_r)

    # zero-mean signals
    grid_a_z = grid_a - np.mean(grid_a)
    grid_r_z = grid_r - np.mean(grid_r)

    # normalize to unit variance to compute a normalized cross-correlation
    std_a = np.std(grid_a_z)
    std_r = np.std(grid_r_z)
    if std_a > 0:
        grid_a_z = grid_a_z / std_a
    if std_r > 0:
        grid_r_z = grid_r_z / std_r

    # compute full cross-correlation (normalized signals)
    corr = np.correlate(grid_a_z, grid_r_z, mode='full')

    # Parabolic interpolation around a peak gives a fractional-sample
    # estimate of the true peak (better than integer-sample argmax).
    def parabolic_peak(c, k):
        if k <= 0 or k >= len(c) - 1:
            return float(k)
        y0, y1, y2 = c[k-1], c[k], c[k+1]
        denom = (y0 - 2*y1 + y2)
        if denom == 0:
            return float(k)
        delta = 0.5 * (y0 - y2) / denom
        return float(k) + delta

    # Convert correlation indices to lag seconds for all possible positions
    num_r = len(grid_r)
    num_a = len(grid_a)
    # lags run from -(num_r-1) .. (num_a-1)
    lag_indices = np.arange(-(num_r - 1), (num_a))
    lag_seconds_all = lag_indices * dt

    # Determine allowable maximum shift: 75% of reference time span
    ref_span = float(t_r.max() - t_r.min()) if t_r.size > 1 else 0.0
    max_allowed_shift = 0.75 * ref_span

    # If reference span is zero (degenerate), allow only very small shifts (one dt)
    if max_allowed_shift <= 0:
        max_allowed_shift = dt

    # Consider peaks in descending order of correlation magnitude, pick first
    # peak whose lag is within allowed bounds. If none found, fall back to no shift.
    corr_indices_sorted = np.argsort(corr)[::-1]  # indices into corr sorted by value desc
    chosen_idx_frac = None
    chosen_lag_seconds = None

    for idx in corr_indices_sorted:
        # integer lag corresponding to this correlation index
        lag_samples = idx - (num_r - 1)
        lag_sec = float(lag_samples * dt)
        # Accept if lag does not move the entire array more than allowed
        if abs(lag_sec) <= max_allowed_shift:
            # refine peak with parabolic interpolation
            idx_frac = parabolic_peak(corr, int(idx))
            lag_frac_samples = idx_frac - (num_r - 1)
            lag_sec_frac = float(lag_frac_samples * dt)
            # final check on fractional lag
            if abs(lag_sec_frac) <= max_allowed_shift:
                chosen_idx_frac = idx_frac
                chosen_lag_seconds = lag_sec_frac
                break

    if chosen_idx_frac is None:
        # No acceptable peak found: do not shift (safer than aligning totally off)
        return times

    # shift original times so array would be aligned with ref_array
    times_shifted = times - float(chosen_lag_seconds)
    return times_shifted

def clean_arrays(t1, v1, t2, v2):
    """Remove any duplicate values and ensure strictly increasing time arrays."""
    def clean_single(t, v):
        sort_idx = np.argsort(t)
        t_s = t[sort_idx]
        v_s = v[sort_idx]
        unique_t, unique_idx = np.unique(t_s, return_index=True)
        t_c = unique_t
        v_c = v_s[unique_idx]
        return t_c, v_c

    t1_c, v1_c = clean_single(t1, v1)
    t2_c, v2_c = clean_single(t2, v2)
    return t1_c, v1_c, t2_c, v2_c

def add_padding(arr1_t, arr1_v, arr2_t, arr2_v):
    """Add padding points to the beginning and end of both arrays
    so they cover the same time span.
    Time locations are kept the same; the first/last values are extended
    as needed to match the length on the end of the other array.
    Assumes arrays of time are ordered.
    """

    # get and compare values of time betweeen arrays
    # if value is smaller (arary is shorter)
    #     array1t add datapoint with valuet+1, valuey
    #     array1v

    # ave1 = get_average_time_step(arr1_t) # this is creating too large of step descripancy between parametrics
    # ave2 = get_average_time_step(arr2_t) # this is creating too large of step descripancy between parametrics
    ave1 = 1 / 60  # assume 60 Hz for reference
    ave2 = 1 / 60  # assume 60 Hz for reference
    while arr1_t[-1] < arr2_t[-1]:
        arr1_t = np.append(arr1_t, arr1_t[-1]+ave1)
        arr1_v = np.append(arr1_v, arr1_v[-1])
    while arr2_t[-1] < arr1_t[-1]:
        arr2_t = np.append(arr2_t, arr2_t[-1]+ave2)
        arr2_v = np.append(arr2_v, arr2_v[-1])
    while arr1_t[0] > arr2_t[0]:
        arr1_t = np.insert(arr1_t, 0, arr1_t[0]-ave1)
        arr1_v = np.insert(arr1_v, 0, arr1_v[0])
    while arr2_t[0] > arr1_t[0]:
        arr2_t = np.insert(arr2_t, 0, arr2_t[0]-ave2)
        arr2_v = np.insert(arr2_v, 0, arr2_v[0])

    # Add dummy values to ensure arrays are of same length without affecting time span
    while arr1_t.size < arr2_t.size:
        arr1_t = np.append(arr1_t, arr1_t[-1])
        arr1_v = np.append(arr1_v, arr1_v[-1])
    while arr2_t.size < arr1_t.size:
        arr2_t = np.append(arr2_t, arr2_t[-1])
        arr2_v = np.append(arr2_v, arr2_v[-1])

    return arr1_t, arr1_v, arr2_t, arr2_v

def normalize_error(error):
    """Returns value scaled from 0 to 1 based on equation x/(x+1), 
    where original error is a positive real value.
    More sensitive to small errors ~0, saturates towards 1 for large errors."""
    return (9 + (error-0.9) / (error-0.9+1)) / 10

def character_error(character, reference_character, ref_ascii, orig_char):
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

    # align char_times to ref_times to account for possible time offset
    char_times_x = adjust_time_delay(char_times, char_x, ref_times, ref_x)
    char_times_y = adjust_time_delay(char_times, char_y, ref_times, ref_y)
    char_times_x, char_x, ref_times, ref_x = clean_arrays(char_times_x, char_x, ref_times, ref_x)
    char_times_y, char_y, ref_times, ref_y = clean_arrays(char_times_y, char_y, ref_times, ref_y)
    char_times_x, char_x, ref_times_x, ref_x = add_padding(char_times_x, char_x, ref_times, ref_x)
    char_times_y, char_y, ref_times_y, ref_y = add_padding(char_times_y, char_y, ref_times, ref_y)

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
            print('fall back to numpy interp')
            return lambda tt: np.interp(np.asarray(tt, dtype=float), t_s, v_s, left=v_s[0], right=v_s[-1])

    sx = build_interpolant(char_times_x, char_x)
    sy = build_interpolant(char_times_y, char_y)
    if sx is None or sy is None:
        return (float('inf'), float('inf'))

    # store adjusted splines for visualization
    if all_character_adjusted_splines.get(orig_char.__hash__(), None) is None:
        all_character_adjusted_splines[orig_char.__hash__()] = {}
    all_character_adjusted_splines[orig_char.__hash__()][ref_ascii] = {
        'x_values': char_x,
        'y_values': char_y,
        'x_times': char_times_x,
        'y_times': char_times_y
    }

    # evaluate interpolants at character timestamps
    try:
        pred_x = sx(ref_times_x)
    except Exception:
        pred_x = np.asarray([sx(t) for t in char_times], dtype=float)
    try:
        pred_y = sy(ref_times_y)
    except Exception:
        pred_y = np.asarray([sy(t) for t in char_times], dtype=float)

    # Mean squared error
    mse_x = float(np.mean((ref_x - pred_x) ** 2))
    mse_y = float(np.mean((ref_y - pred_y) ** 2))

    return (mse_x, mse_y)

def identify_character(character: Character, display=True):
    """
    Analyze the character and attempt to identify it.
    Returns a dictionary with key for each character and a confidence score (0-1).
    """
    d = {}
    for ascii_char, ref_char in REFERENCE_CHARACTERS.items():
        x_error, y_error = character_error(trim_leading_time(normalize_positions(character)), 
                                           trim_leading_time(normalize_positions(ref_char)), ascii_char, character)
        average_error = (x_error + y_error) / 2
        confidence = 1 - normalize_error(average_error)
        d[ascii_char] = confidence
    if display:
        print("---- Character Results: ---")
        for k in sorted(d.keys()):
            print(f"  '{k}': {d[k]:.8f}")
        print("--------------------------\n")
    return d

def identify_screen_characters(screen_character: Character, threshold=0.7):
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
        print("Analyzing indivisual stroke index: ", i)
        sub_char = create_sub_character(screen_character, [i])
        result = identify_character(sub_char)
        if result:
            # For simplicity, assume result is a dict {char: confidence}
            best_match = max(result.items(), key=lambda kv: kv[1])
            if best_match[1] >= threshold:  # confidence threshold
                identified.append((best_match[0], best_match[1], sub_char))
                print(f"Identified '{best_match[0]}' with confidence {best_match[1]:.2f}")

    for i in range(0, len(screen_character.strokes)-1):
        print("Analyzing two-stroke combination indices: ", i, i+1)
        if i+1 >= len(screen_character.strokes):
            break
        sub_char = create_sub_character(screen_character, [i, i+1])
        result = identify_character(sub_char)
        if result:
            best_match = max(result.items(), key=lambda kv: kv[1])
            if best_match[1] >= threshold:
                identified.append((best_match[0], best_match[1], sub_char))
                print(f"Identified '{best_match[0]}' with confidence {best_match[1]:.2f}")
    
    for i in range(0, len(screen_character.strokes)-2):
        print("Analyzing three-stroke combination indices: ", i, i+1, i+2)
        if i+2 >= len(screen_character.strokes):
            break
        sub_char = create_sub_character(screen_character, [i, i+1, i+2])
        result = identify_character(sub_char)
        if result:
            best_match = max(result.items(), key=lambda kv: kv[1])
            if best_match[1] >= threshold:
                identified.append((best_match[0], best_match[1], sub_char))
                print(f"Identified '{best_match[0]}' with confidence {best_match[1]:.2f}")

    return identified