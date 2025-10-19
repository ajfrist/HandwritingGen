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
def visualize_strokes(character, include_refs=False):
    """
    Create a Plotly 3D visualization showing x (0-1), y (0-1) and time (s).
    If include_refs=True, overlay all reference characters (hidden by default).
    Also adds a synced duplicate of each reference aligned to the current character.
    """
    if not character or len(character) == 0:
        print("No strokes to visualize.")
        return

    if go is None or px is None or plotly_offline_plot is None:
        print("Plotly is not available. Install it with: pip install plotly")
        return

    traces = []
    palette = px.colors.qualitative.Plotly

    # Current character (trim+normalize for consistent comparison)
    cur_norm = normalize_positions(trim_leading_time(character))
    for i, stroke in enumerate(cur_norm):
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
                name=f"current stroke {i+1}",
                visible=True
            )
        )

    # Collect numeric ranges from current character
    xs_all = [p.x_norm for p in cur_norm.all_points()]
    ys_all = [p.y_norm for p in cur_norm.all_points()]
    ts_all = [p.timestamp for p in cur_norm.all_points()]

    # Add reference characters: both original (legendonly) and synced duplicate (legendonly)
    if include_refs and REFERENCE_CHARACTERS:
        # flatten current arrays used for alignment
        cur_pts = list(cur_norm.all_points())
        if cur_pts:
            cur_times = np.array([p.timestamp for p in cur_pts], dtype=float)
            cur_x = np.array([p.x_norm for p in cur_pts], dtype=float)
            cur_y = np.array([p.y_norm for p in cur_pts], dtype=float)
        else:
            cur_times = np.array([], dtype=float)
            cur_x = np.array([], dtype=float)
            cur_y = np.array([], dtype=float)

        for key, ref in REFERENCE_CHARACTERS.items():
            ref_norm = normalize_positions(trim_leading_time(ref))

            # Build flat lists for reference (no separators) for alignment
            ref_times_flat = []
            ref_x_flat = []
            ref_y_flat = []
            # Also build combined lists with separators for original plotting
            xs = []
            ys = []
            zs = []
            for stroke in ref_norm:
                if len(stroke) == 0:
                    continue
                for p in stroke:
                    ref_times_flat.append(p.timestamp)
                    ref_x_flat.append(p.x_norm)
                    ref_y_flat.append(p.y_norm)
                    xs.append(p.x_norm)
                    ys.append(p.y_norm)
                    zs.append(p.timestamp)
                # separator between strokes
                xs.append(None)
                ys.append(None)
                zs.append(None)

            if not ref_times_flat:
                continue

            # original reference trace (hidden by default)
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode='lines',
                    line=dict(color='lightgray', width=3),
                    name=f"ref {key} (orig)",
                    visible='legendonly',
                    hoverinfo='skip'
                )
            )

            # attempt to compute synced times for ref to align with current
            try:
                shifted_ref_times = adjust_time_delay(np.asarray(ref_times_flat, dtype=float),
                                                      np.asarray(ref_x_flat, dtype=float),
                                                      cur_x, ref_times=cur_times)
                # adjust_time_delay shifts the 'times' passed; but in our call we passed ref_times as 'times',
                # array=ref_x_flat, ref_array=cur_x -> shifted_ref_times aligns ref -> cur
                # shifted_ref_times is same length as ref_times_flat
                # reconstruct synced zs by inserting None separators between strokes
                synced_zs = []
                it = iter(shifted_ref_times)
                for stroke in ref_norm:
                    if len(stroke) == 0:
                        continue
                    for _ in stroke:
                        synced_zs.append(next(it))
                    synced_zs.append(None)
                # synced trace (hidden by default) using same xs/ys but synced zs
                traces.append(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=synced_zs,
                        mode='lines',
                        line=dict(color='royalblue', width=2, dash='dash'),
                        name=f"ref {key} (synced)",
                        visible='legendonly',
                        hoverinfo='skip'
                    )
                )
                # extend numeric ranges
                xs_all.extend([v for v in xs if v is not None])
                ys_all.extend([v for v in ys if v is not None])
                ts_all.extend([v for v in zs if v is not None])
                # include synced times too
                ts_all.extend([v for v in synced_zs if v is not None])
            except Exception:
                # fallback: still include original numeric ranges
                xs_all.extend([v for v in xs if v is not None])
                ys_all.extend([v for v in ys if v is not None])
                ts_all.extend([v for v in zs if v is not None])

    # Determine global ranges and add padding
    if xs_all and ys_all and ts_all:
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        t_min, t_max = min(ts_all), max(ts_all)
    else:
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
        t_min, t_max = 0.0, 1.0

    def pad_range(a_min, a_max, frac=0.05, fixed=0.05):
        span = a_max - a_min
        if span == 0:
            p = fixed
        else:
            p = span * frac
        return (a_min - p, a_max + p)

    xr = pad_range(x_min, x_max)
    yr = pad_range(y_min, y_max)
    zr = pad_range(t_min, t_max)

    layout = go.Layout(
        title="Touch strokes: X, Y over time (current + references)",
        scene=dict(
            xaxis=dict(title="X (normalized)", range=[xr[0], xr[1]]),
            yaxis=dict(title="Y (normalized)", range=[yr[0], yr[1]]),
            zaxis=dict(title="Time (s)", range=[max(0, zr[0]), zr[1]]),
        ),
        legend=dict(itemsizing='constant')
    )

    fig = go.Figure(data=traces, layout=layout)
    try:
        plotly_offline_plot(fig, auto_open=True, filename='handwriting_3d_comparison.html')
    except Exception as e:
        try:
            fig.show(renderer='browser')
        except Exception:
            print("Unable to open Plotly visualization:", e)


def visualize_parametric(character, include_refs=False):
    """
    Create two Plotly 2D visualizations (X(t) and Y(t)) with raw data and
    interpolated curves. If include_refs=True, overlay reference characters
    (hidden by default) for selective comparison. Also adds synced duplicate
    of each reference shifted to best overlap the current character.
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

    # Normalize & trim current character for consistent comparison
    cur_norm = normalize_positions(trim_leading_time(character))
    all_points = list(cur_norm.all_points())
    if not all_points:
        print("No data points for parametric visualization.")
        return

    all_times = np.array([p.timestamp for p in all_points])
    all_x = np.array([p.x_norm for p in all_points])
    all_y = np.array([p.y_norm for p in all_points])

    # sort by time
    order = np.argsort(all_times)
    all_times = all_times[order]
    all_x = all_x[order]
    all_y = all_y[order]

    t_smooth = np.linspace(all_times.min(), all_times.max(), 500)

    # interpolation for current character
    newton_x = newton_interpolation(all_times, all_x.copy())
    newton_y = newton_interpolation(all_times, all_y.copy())
    spline_x = cubic_spline_interpolation(all_times, all_x)
    spline_y = cubic_spline_interpolation(all_times, all_y)

    traces_x.extend([
        go.Scatter(
            x=t_smooth, y=[newton_x(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(255,0,0,0.8)', width=3),
            name='current - Newton', visible=True
        ),
        go.Scatter(
            x=t_smooth, y=[spline_x(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(0,200,0,0.8)', width=3),
            name='current - Spline', visible=True
        )
    ])

    traces_y.extend([
        go.Scatter(
            x=t_smooth, y=[newton_y(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(255,0,0,0.8)', width=3),
            name='current - Newton', visible=True
        ),
        go.Scatter(
            x=t_smooth, y=[spline_y(t) for t in t_smooth],
            mode='lines', line=dict(color='rgba(0,200,0,0.8)', width=3),
            name='current - Spline', visible=True
        )
    ])

    # Add original data points for current char grouped by stroke
    for i, stroke in enumerate(cur_norm):
        if len(stroke) == 0:
            continue
        ts = [p.timestamp for p in stroke]
        xs = [p.x_norm for p in stroke]
        ys = [p.y_norm for p in stroke]
        color = palette[i % len(palette)]
        traces_x.append(go.Scatter(x=ts, y=xs, mode='markers', marker=dict(size=6, color=color),
                                   name=f"current stroke {i+1} (data)", visible=True))
        traces_y.append(go.Scatter(x=ts, y=ys, mode='markers', marker=dict(size=6, color=color),
                                   name=f"current stroke {i+1} (data)", visible=True))

    # Collect global ranges starting with current data
    x_vals = list(all_x)
    y_vals = list(all_y)
    t_vals = list(all_times)

    # Reference characters: add original and synced versions
    if include_refs and REFERENCE_CHARACTERS:
        for key, ref in REFERENCE_CHARACTERS.items():
            ref_norm = normalize_positions(trim_leading_time(ref))
            ref_pts = list(ref_norm.all_points())
            if not ref_pts:
                continue
            r_times = np.array([p.timestamp for p in ref_pts], dtype=float)
            r_x = np.array([p.x_norm for p in ref_pts], dtype=float)
            r_y = np.array([p.y_norm for p in ref_pts], dtype=float)

            # sort & unique times
            idx = np.argsort(r_times)
            r_times = r_times[idx]
            r_x = r_x[idx]
            r_y = r_y[idx]

            # Build dense t_ref and evaluate ref interpolant
            if r_times.size == 1:
                t_ref = np.linspace(0.0, r_times[0] or 1.0, 200)
                xr = np.full_like(t_ref, r_x[0])
                yr = np.full_like(t_ref, r_y[0])
            else:
                t_ref = np.linspace(0.0, r_times.max(), 200)
                try:
                    fx = cubic_spline_interpolation(r_times, r_x)
                    fy = cubic_spline_interpolation(r_times, r_y)
                    xr = fx(t_ref)
                    yr = fy(t_ref)
                except Exception:
                    xr = np.interp(t_ref, r_times, r_x, left=r_x[0], right=r_x[-1])
                    yr = np.interp(t_ref, r_times, r_y, left=r_y[0], right=r_y[-1])

            # original ref traces (hidden)
            traces_x.append(go.Scatter(x=t_ref, y=xr, mode='lines',
                                       line=dict(color='lightgray', width=2),
                                       name=f"ref {key} (orig)", visible='legendonly', hoverinfo='skip'))
            traces_y.append(go.Scatter(x=t_ref, y=yr, mode='lines',
                                       line=dict(color='lightgray', width=2),
                                       name=f"ref {key} (orig)", visible='legendonly', hoverinfo='skip'))

            # attempt to compute sync shift (align ref points to current points)
            try:
                shifted_r_times = adjust_time_delay(r_times, r_x, all_x, ref_times=all_times)
                # compute average shift (shifted - original)
                shift_vals = shifted_r_times - r_times
                shift_seconds = float(np.median(shift_vals)) if shift_vals.size > 0 else 0.0
                # apply same shift to dense t_ref grid
                t_ref_synced = t_ref + shift_seconds

                traces_x.append(go.Scatter(x=t_ref_synced, y=xr, mode='lines',
                                           line=dict(color='royalblue', width=1, dash='dash'),
                                           name=f"ref {key} (synced)", visible='legendonly', hoverinfo='skip'))
                traces_y.append(go.Scatter(x=t_ref_synced, y=yr, mode='lines',
                                           line=dict(color='royalblue', width=1, dash='dash'),
                                           name=f"ref {key} (synced)", visible='legendonly', hoverinfo='skip'))

                # extend global ranges with both original and synced
                x_vals.extend(list(xr))
                y_vals.extend(list(yr))
                t_vals.extend(list(t_ref))
                t_vals.extend(list(t_ref_synced))
            except Exception:
                x_vals.extend(list(xr))
                y_vals.extend(list(yr))
                t_vals.extend(list(t_ref))

    # compute ranges and padding
    if x_vals and y_vals and t_vals:
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        t_min, t_max = min(t_vals), max(t_vals)
    else:
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0
        t_min, t_max = 0.0, 1.0

    def pad_range(a_min, a_max, frac=0.05, fixed=0.05):
        span = a_max - a_min
        if span == 0:
            p = fixed
        else:
            p = span * frac
        return (a_min - p, a_max + p)

    xr = pad_range(x_min, x_max)
    yr = pad_range(y_min, y_max)
    tr = pad_range(t_min, t_max)

    layout_x = go.Layout(
        title="X over Time (current + references)",
        xaxis=dict(title="Time (s)", range=[max(0, tr[0]), tr[1]]),
        yaxis=dict(title="X (normalized)", range=[xr[0], xr[1]]),
    )
    layout_y = go.Layout(
        title="Y over Time (current + references)",
        xaxis=dict(title="Time (s)", range=[max(0, tr[0]), tr[1]]),
        yaxis=dict(title="Y (normalized)", range=[yr[0], yr[1]]),
    )

    fig_x = go.Figure(data=traces_x, layout=layout_x)
    fig_y = go.Figure(data=traces_y, layout=layout_y)

    try:
        plotly_offline_plot(fig_x, auto_open=True, filename='handwriting_x_vs_t_comparison.html')
    except Exception:
        try:
            fig_x.show(renderer='browser')
        except Exception as e:
            print("Unable to open X(t) plot:", e)

    try:
        plotly_offline_plot(fig_y, auto_open=True, filename='handwriting_y_vs_t_comparison.html')
    except Exception:
        try:
            fig_y.show(renderer='browser')
        except Exception as e:
            print("Unable to open Y(t) plot:", e)

def visualize_comparison(character, best_key, s_key='s'):
    """
    Open 3 plots (3D, X(t), Y(t)) comparing:
      - current (orig)
      - best reference (orig)
      - 's' reference (orig)
      - current aligned to best reference
      - current aligned to 's' reference
    All reference traces are hidden by default (legendonly) so user can toggle visibility.
    """
    if not character or len(character) == 0:
        print("No current character to compare.")
        return

    if go is None or px is None or plotly_offline_plot is None:
        print("Plotly not available.")
        return

    # Normalize & trim current
    cur = normalize_positions(trim_leading_time(character))
    cur_pts = list(cur.all_points())
    if not cur_pts:
        print("No points in current character.")
        return
    cur_times = np.array([p.timestamp for p in cur_pts], dtype=float)
    cur_x = np.array([p.x_norm for p in cur_pts], dtype=float)
    cur_y = np.array([p.y_norm for p in cur_pts], dtype=float)
    # build flat xy/z with None separators for strokes
    cur_xs = []
    cur_ys = []
    cur_zs = []
    for stroke in cur:
        for p in stroke:
            cur_xs.append(p.x_norm)
            cur_ys.append(p.y_norm)
            cur_zs.append(p.timestamp)
        cur_xs.append(None); cur_ys.append(None); cur_zs.append(None)

    # helper to prepare reference dense curves
    def prepare_ref(key):
        ref = REFERENCE_CHARACTERS.get(key)
        if ref is None:
            return None
        refn = normalize_positions(trim_leading_time(ref))
        ref_pts = list(refn.all_points())
        if not ref_pts:
            return None
        rt = np.array([p.timestamp for p in ref_pts], dtype=float)
        rx = np.array([p.x_norm for p in ref_pts], dtype=float)
        ry = np.array([p.y_norm for p in ref_pts], dtype=float)
        # dense parameter t_ref
        if rt.size == 1:
            t_ref = np.linspace(0.0, rt[0] or 1.0, 300)
            xr = np.full_like(t_ref, rx[0])
            yr = np.full_like(t_ref, ry[0])
        else:
            t_ref = np.linspace(0.0, rt.max(), 300)
            try:
                fx = cubic_spline_interpolation(rt, rx)
                fy = cubic_spline_interpolation(rt, ry)
                xr = fx(t_ref)
                yr = fy(t_ref)
            except Exception:
                xr = np.interp(t_ref, rt, rx, left=rx[0], right=rx[-1])
                yr = np.interp(t_ref, rt, ry, left=ry[0], right=ry[-1])

        # also build flat original lists with separators for 3D plotting
        xs = []; ys = []; zs = []
        for stroke in refn:
            for p in stroke:
                xs.append(p.x_norm); ys.append(p.y_norm); zs.append(p.timestamp)
            xs.append(None); ys.append(None); zs.append(None)
        return dict(rt=rt, rx=rx, ry=ry, t_ref=t_ref, xr=xr, yr=yr, xs=xs, ys=ys, zs=zs)

    best_ref = prepare_ref(best_key) if best_key else None
    s_ref = prepare_ref(s_key)

    # compute aligned times of current to each reference (based on x-series)
    def align_current_to_ref(ref):
        if ref is None:
            return None, 0.0
        try:
            shifted_times = adjust_time_delay(cur_times, cur_x, ref['rx'], ref_times=ref['rt'])
            # lag = median(original - shifted)
            lag = float(np.median(cur_times - shifted_times))
            # Build synced zs for 3D (map each original point to shifted timestamp)
            synced_zs = []
            it = iter(shifted_times)
            for stroke in cur:
                for _ in stroke:
                    synced_zs.append(next(it))
                synced_zs.append(None)
            return np.asarray(synced_zs, dtype=float), lag
        except Exception:
            return None, 0.0

    synced_z_best, lag_best = align_current_to_ref(best_ref)
    synced_z_s, lag_s = align_current_to_ref(s_ref)

    # Build 3D traces (5 traces)
    traces3d = []
    # current original
    traces3d.append(go.Scatter3d(x=cur_xs, y=cur_ys, z=cur_zs,
                                 mode='lines+markers',
                                 line=dict(color='red', width=4),
                                 marker=dict(size=3, color='red'),
                                 name='current (orig)', visible=True))
    # best ref original
    if best_ref:
        traces3d.append(go.Scatter3d(x=best_ref['xs'], y=best_ref['ys'], z=best_ref['zs'],
                                     mode='lines', line=dict(color='lightgray', width=2),
                                     name=f"ref {best_key} (orig)", visible='legendonly', hoverinfo='skip'))
    else:
        traces3d.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name="ref (missing)", visible='legendonly'))
    # s ref original
    if s_ref:
        traces3d.append(go.Scatter3d(x=s_ref['xs'], y=s_ref['ys'], z=s_ref['zs'],
                                     mode='lines', line=dict(color='lightgray', width=2),
                                     name=f"ref {s_key} (orig)", visible='legendonly', hoverinfo='skip'))
    else:
        traces3d.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name="s ref (missing)", visible='legendonly'))
    # current aligned to best
    if synced_z_best is not None:
        traces3d.append(go.Scatter3d(x=cur_xs, y=cur_ys, z=synced_z_best,
                                     mode='lines', line=dict(color='blue', width=2, dash='dash'),
                                     name=f"current aligned -> {best_key}", visible='legendonly'))
    else:
        traces3d.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name="aligned best (missing)", visible='legendonly'))
    # current aligned to s
    if synced_z_s is not None:
        traces3d.append(go.Scatter3d(x=cur_xs, y=cur_ys, z=synced_z_s,
                                     mode='lines', line=dict(color='green', width=2, dash='dash'),
                                     name=f"current aligned -> {s_key}", visible='legendonly'))
    else:
        traces3d.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name="aligned s (missing)", visible='legendonly'))

    # determine global ranges
    all_x_vals = [v for v in cur_xs if v is not None]
    all_y_vals = [v for v in cur_ys if v is not None]
    all_t_vals = [v for v in cur_zs if v is not None]
    if best_ref:
        all_x_vals.extend([v for v in best_ref['xs'] if v is not None])
        all_y_vals.extend([v for v in best_ref['ys'] if v is not None])
        all_t_vals.extend([v for v in best_ref['zs'] if v is not None])
    if s_ref:
        all_x_vals.extend([v for v in s_ref['xs'] if v is not None])
        all_y_vals.extend([v for v in s_ref['ys'] if v is not None])
        all_t_vals.extend([v for v in s_ref['zs'] if v is not None])
    # include synced times
    if synced_z_best is not None:
        all_t_vals.extend([v for v in synced_z_best if v is not None])
    if synced_z_s is not None:
        all_t_vals.extend([v for v in synced_z_s if v is not None])

    def pad(a_min, a_max):
        span = a_max - a_min
        if span == 0:
            p = 0.05
        else:
            p = span * 0.05
        return a_min - p, a_max + p

    if all_x_vals and all_y_vals and all_t_vals:
        xmin, xmax = min(all_x_vals), max(all_x_vals)
        ymin, ymax = min(all_y_vals), max(all_y_vals)
        tmin, tmax = min(all_t_vals), max(all_t_vals)
    else:
        xmin, xmax, ymin, ymax, tmin, tmax = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

    xr = pad(xmin, xmax)
    yr = pad(ymin, ymax)
    tr = pad(tmin, tmax)

    layout3d = go.Layout(title=f"3D comparison (current + {best_key} + {s_key})",
                         scene=dict(xaxis=dict(title="X", range=[xr[0], xr[1]]),
                                    yaxis=dict(title="Y", range=[yr[0], yr[1]]),
                                    zaxis=dict(title="Time", range=[max(0, tr[0]), tr[1]])),
                         legend=dict(itemsizing='constant'))

    fig3d = go.Figure(data=traces3d, layout=layout3d)
    try:
        plotly_offline_plot(fig3d, auto_open=True, filename='comparison_3d.html')
    except Exception:
        try:
            fig3d.show(renderer='browser')
        except Exception:
            print("Unable to open 3D comparison")
    # Prepare parametric plots (X(t), Y(t)) using the already-computed
    # cur_times, cur_x, cur_y arrays above. Ensure they are sorted by time
    # before building interpolants so no NameError / undefined-variable
    # issues occur (previous code referenced undefined all_times/all_x/all_y).
    try:
        order = np.argsort(cur_times)
        cur_times = np.asarray(cur_times, dtype=float)[order]
        cur_x = np.asarray(cur_x, dtype=float)[order]
        cur_y = np.asarray(cur_y, dtype=float)[order]
    except Exception:
        # Fallback: build from normalized character points
        cur_norm = normalize_positions(trim_leading_time(character))
        all_points = list(cur_norm.all_points())
        cur_times = np.array([p.timestamp for p in all_points], dtype=float)
        cur_x = np.array([p.x_norm for p in all_points], dtype=float)
        cur_y = np.array([p.y_norm for p in all_points], dtype=float)
        order = np.argsort(cur_times) if cur_times.size else np.array([], dtype=int)
        if order.size:
            cur_times = cur_times[order]
            cur_x = cur_x[order]
            cur_y = cur_y[order]

    # dense time grid for plotting current interpolants
    if cur_times.size > 1:
        t_min_cur, t_max_cur = cur_times.min(), cur_times.max()
        t_smooth = np.linspace(t_min_cur, t_max_cur, 500)
    else:
        t_smooth = np.linspace(0.0, 1.0, 500)

    # optional Newton (not required later) left out; build spline/interpolants below
    # --- Parametric X(t) and Y(t): build 5-trace plots each ---
    # build dense time grid for current
    cur_times_sorted = np.array(sorted(cur_times))
    if cur_times_sorted.size > 1:
        t_min_cur, t_max_cur = cur_times_sorted.min(), cur_times_sorted.max()
        t_smooth = np.linspace(t_min_cur, t_max_cur, 500)
    else:
        t_smooth = np.linspace(0.0, 1.0, 500)

    # current interpolants
    try:
        fx_cur = cubic_spline_interpolation(cur_times, cur_x)
        fy_cur = cubic_spline_interpolation(cur_times, cur_y)
    except Exception:
        # fallback to simple numpy interp wrappers
        fx_cur = lambda tt: np.interp(np.asarray(tt, dtype=float), cur_times, cur_x, left=cur_x[0], right=cur_x[-1])
        fy_cur = lambda tt: np.interp(np.asarray(tt, dtype=float), cur_times, cur_y, left=cur_y[0], right=cur_y[-1])

    # prepare reference dense curves (if present)
    def dense_ref(ref):
        if ref is None:
            return None
        t_ref = ref['t_ref']
        xr = ref['xr']
        yr = ref['yr']
        return dict(t_ref=t_ref, xr=xr, yr=yr)

    best_dense = dense_ref(best_ref)
    s_dense = dense_ref(s_ref)

    # compute lags (median) for shifting current to refs
    def compute_lag_to_ref(ref):
        if ref is None:
            return 0.0
        # align current (cur_times, cur_x) to reference sample points (ref['rt'], ref['rx'])
        try:
            shifted = adjust_time_delay(cur_times, cur_x, ref['rx'], ref_times=ref['rt'])
            lag = float(np.median(cur_times - shifted))
            return lag
        except Exception:
            return 0.0

    lag_best = compute_lag_to_ref(best_ref)
    lag_s = compute_lag_to_ref(s_ref)

    # build traces for X(t)
    traces_x = []
    # current (spline)
    traces_x.append(go.Scatter(x=t_smooth, y=fx_cur(t_smooth),
                               mode='lines', line=dict(color='red', width=3),
                               name='current (spline)', visible=True))
    # best ref orig
    if best_dense:
        traces_x.append(go.Scatter(x=best_dense['t_ref'], y=best_dense['xr'],
                                   mode='lines', line=dict(color='lightgray', width=2),
                                   name=f"ref {best_key} (orig)", visible='legendonly'))
    else:
        traces_x.append(go.Scatter(x=[], y=[], mode='lines', name=f"ref {best_key} (missing)", visible='legendonly'))
    # s ref orig
    if s_dense:
        traces_x.append(go.Scatter(x=s_dense['t_ref'], y=s_dense['xr'],
                                   mode='lines', line=dict(color='lightgray', width=2),
                                   name=f"ref {s_key} (orig)", visible='legendonly'))
    else:
        traces_x.append(go.Scatter(x=[], y=[], mode='lines', name=f"ref {s_key} (missing)", visible='legendonly'))
    # current aligned -> best (shifted by lag_best)
    traces_x.append(go.Scatter(x=(t_smooth - lag_best), y=fx_cur(t_smooth),
                               mode='lines', line=dict(color='blue', width=2, dash='dash'),
                               name=f"current aligned -> {best_key}", visible='legendonly'))
    # current aligned -> s
    traces_x.append(go.Scatter(x=(t_smooth - lag_s), y=fx_cur(t_smooth),
                               mode='lines', line=dict(color='green', width=2, dash='dash'),
                               name=f"current aligned -> {s_key}", visible='legendonly'))

    # build traces for Y(t)
    traces_y = []
    traces_y.append(go.Scatter(x=t_smooth, y=fy_cur(t_smooth),
                               mode='lines', line=dict(color='red', width=3),
                               name='current (spline)', visible=True))
    if best_dense:
        traces_y.append(go.Scatter(x=best_dense['t_ref'], y=best_dense['yr'],
                                   mode='lines', line=dict(color='lightgray', width=2),
                                   name=f"ref {best_key} (orig)", visible='legendonly'))
    else:
        traces_y.append(go.Scatter(x=[], y=[], mode='lines', name=f"ref {best_key} (missing)", visible='legendonly'))
    if s_dense:
        traces_y.append(go.Scatter(x=s_dense['t_ref'], y=s_dense['yr'],
                                   mode='lines', line=dict(color='lightgray', width=2),
                                   name=f"ref {s_key} (orig)", visible='legendonly'))
    else:
        traces_y.append(go.Scatter(x=[], y=[], mode='lines', name=f"ref {s_key} (missing)", visible='legendonly'))

    traces_y.append(go.Scatter(x=(t_smooth - lag_best), y=fy_cur(t_smooth),
                               mode='lines', line=dict(color='blue', width=2, dash='dash'),
                               name=f"current aligned -> {best_key}", visible='legendonly'))
    traces_y.append(go.Scatter(x=(t_smooth - lag_s), y=fy_cur(t_smooth),
                               mode='lines', line=dict(color='green', width=2, dash='dash'),
                               name=f"current aligned -> {s_key}", visible='legendonly'))

    # set axis ranges using previously computed tr, xr, yr (from 3D section)
    layout_x = go.Layout(title=f"X(t) comparison (current + {best_key} + {s_key})",
                         xaxis=dict(title="Time (s)", range=[max(0, tr[0]), tr[1]]),
                         yaxis=dict(title="X (normalized)", range=[xr[0], xr[1]]))
    layout_y = go.Layout(title=f"Y(t) comparison (current + {best_key} + {s_key})",
                         xaxis=dict(title="Time (s)", range=[max(0, tr[0]), tr[1]]),
                         yaxis=dict(title="Y (normalized)", range=[yr[0], yr[1]]))

    fig_x = go.Figure(data=traces_x, layout=layout_x)
    fig_y = go.Figure(data=traces_y, layout=layout_y)

    try:
        plotly_offline_plot(fig_x, auto_open=True, filename='comparison_x.html')
    except Exception:
        try:
            fig_x.show(renderer='browser')
        except Exception:
            pass
    try:
        plotly_offline_plot(fig_y, auto_open=True, filename='comparison_y.html')
    except Exception:
        try:
            fig_y.show(renderer='browser')
        except Exception:
            pass

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

def adjust_time_delay(times, array, ref_array, ref_times=None):
    """
    Align `array` to `ref_array` by finding the time delay that maximizes their
    cross-correlation and returning a new times array shifted by that delay.

    Parameters:
    - times: 1D array of timestamps for `array`
    - array: 1D array of sample values (may contain None or np.nan for gaps)
    - ref_array: 1D array of reference sample values (may contain None/np.nan)
    - ref_times: optional 1D array of timestamps for ref_array; if None an equispaced
      axis will be created for ref_array spanning the same interval as `times`.

    Returns:
    - times_shifted: numpy array same shape as `times`, shifted to best align with ref_array.
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
    idx_max = int(np.argmax(corr))

    # lags run from -(len(grid_r)-1) .. (len(grid_a)-1)
    lag_samples = idx_max - (len(grid_r) - 1)

    # Parabolic interpolation around the peak gives a fractional-sample
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

    idx_frac = parabolic_peak(corr, idx_max)
    lag_frac_samples = idx_frac - (len(grid_r) - 1)
    lag_seconds = float(lag_frac_samples * dt)

    # shift original times so array would be aligned with ref_array
    times_shifted = times - lag_seconds
    return times_shifted

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

    # align char_times to ref_times to account for possible time offset
    char_times = adjust_time_delay(char_times, char_x, ref_x, ref_times)

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
                
                print("Generating visualizations (current + references)...")
                # Temporarily hide pygame window
                pygame.display.iconify()
                
                # Generate visualizations including the 52 reference characters (hidden by default)
                try:
                    visualize_strokes(current_character, include_refs=True)
                    visualize_parametric(current_character, include_refs=True)
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

                # comment if want to compare
                # -----
                # continue
                # -----

                # New: open comparison graphs for best match + 's' reference
                try:
                    # determine best overall from REFERENCE_CHARACTERS using identify_character
                    scores = identify_character(trim_leading_time(normalize_positions(current_character)))
                    if scores:
                        # find best key
                        best_key = max(scores.items(), key=lambda kv: kv[1])[0]
                    else:
                        best_key = None
                    # if 's' not present, will be handled in visualize_comparison
                    visualize_comparison(current_character, best_key, s_key='L')
                except Exception as e:
                    print("Comparison visualization failed:", e)

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
