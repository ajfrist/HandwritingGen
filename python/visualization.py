import numpy as np

from interpolation import newton_interpolation, cubic_spline_interpolation
from character_matching import adjust_time_delay, get_current_adjusted_splines
from data_structures import normalize_positions, trim_leading_time, REFERENCE_CHARACTERS

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot as plotly_offline_plot
except Exception:
    go = None
    px = None
    plotly_offline_plot = None

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
    # Ensure in order
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
    # cur_times_sorted = np.array(sorted(cur_times))
    # t_min_cur, t_max_cur = cur_times_sorted.min(), cur_times_sorted.max()
    # t_smooth = np.linspace(t_min_cur, t_max_cur, 500)

    # Fetch reference character datapoints for graphing
    refn = normalize_positions(trim_leading_time(REFERENCE_CHARACTERS.get(best_key)))
    ref_pts = list(refn.all_points())
    rt_b = np.array([p.timestamp for p in ref_pts], dtype=float)
    rx_b = np.array([p.x_norm for p in ref_pts], dtype=float)
    ry_b = np.array([p.y_norm for p in ref_pts], dtype=float)
    refn = normalize_positions(trim_leading_time(REFERENCE_CHARACTERS.get(s_key)))
    ref_pts = list(refn.all_points())
    rt_s = np.array([p.timestamp for p in ref_pts], dtype=float)
    rx_s = np.array([p.x_norm for p in ref_pts], dtype=float)
    ry_s = np.array([p.y_norm for p in ref_pts], dtype=float)

    # Create data for live calculated adjustments
    character_splines = get_current_adjusted_splines()
    x_values_best = character_splines[character.__hash__()][best_key]['x_values']
    y_values_best = character_splines[character.__hash__()][best_key]['y_values']
    x_times_best = character_splines[character.__hash__()][best_key]['x_times']
    y_times_best = character_splines[character.__hash__()][best_key]['y_times']
    x_values_s = character_splines[character.__hash__()][s_key]['x_values']
    y_values_s = character_splines[character.__hash__()][s_key]['y_values']
    x_times_s = character_splines[character.__hash__()][s_key]['x_times']
    y_times_s = character_splines[character.__hash__()][s_key]['y_times']

    scaled_x_values_best = character_splines[character.__hash__()][best_key+"_scaled"]['x_values']
    scaled_y_values_best = character_splines[character.__hash__()][best_key+"_scaled"]['y_values']
    scaled_x_times_best = character_splines[character.__hash__()][best_key+"_scaled"]['x_times']
    scaled_y_times_best = character_splines[character.__hash__()][best_key+"_scaled"]['y_times']
    scaled_x_values_s = character_splines[character.__hash__()][s_key+"_scaled"]['x_values']
    scaled_y_values_s = character_splines[character.__hash__()][s_key+"_scaled"]['y_values']
    scaled_x_times_s = character_splines[character.__hash__()][s_key+"_scaled"]['x_times']
    scaled_y_times_s = character_splines[character.__hash__()][s_key+"_scaled"]['y_times']


    # build traces for X(t)
    traces_x = []
    # current (spline)
    traces_x.append(go.Scatter(x=cur_times, y=cur_x,
                               mode='lines', line=dict(color='red', width=3),
                               name='current (spline)', visible=True))
    # best ref orig
    traces_x.append(go.Scatter(x=rt_b, y=rx_b,
                                mode='lines', line=dict(color='lightgray', width=2),
                                name=f"ref {best_key} (orig)", visible='legendonly'))
    # s ref orig
    traces_x.append(go.Scatter(x=rt_s, y=rx_s,
                                mode='lines', line=dict(color='lightgray', width=2),
                                name=f"ref {s_key} (orig)", visible='legendonly'))
    # current aligned -> best (shifted by lag_best)
    traces_x.append(go.Scatter(x=x_times_best, y=x_values_best,
                               mode='lines', line=dict(color='blue', width=2, dash='dash'),
                               name=f"current aligned -> {best_key}", visible='legendonly'))
    # current aligned -> s
    traces_x.append(go.Scatter(x=x_times_s, y=x_values_s,
                               mode='lines', line=dict(color='green', width=2, dash='dash'),
                               name=f"current aligned -> {s_key}", visible='legendonly'))
    # current aligned & scaled -> best
    traces_x.append(go.Scatter(x=scaled_x_times_best, y=scaled_x_values_best,
                               mode='lines', line=dict(color='blue', width=1, dash='dot'),
                               name=f"current aligned+scaled -> {best_key}", visible='legendonly'))
    # current aligned & scaled -> s
    traces_x.append(go.Scatter(x=scaled_x_times_s, y=scaled_x_values_s,
                               mode='lines', line=dict(color='green', width=1, dash='dot'),
                               name=f"current aligned+scaled -> {s_key}", visible='legendonly'))

    # build traces for Y(t)
    traces_y = []
    traces_y.append(go.Scatter(x=cur_times, y=cur_y,
                               mode='lines', line=dict(color='red', width=3),
                               name='current (spline)', visible=True))
    traces_y.append(go.Scatter(x=rt_b, y=ry_b,
                               mode='lines', line=dict(color='lightgray', width=2),
                               name=f"ref {best_key} (orig)", visible='legendonly'))
    traces_y.append(go.Scatter(x=rt_s, y=ry_s,
                                mode='lines', line=dict(color='lightgray', width=2),
                                name=f"ref {s_key} (orig)", visible='legendonly'))
    traces_y.append(go.Scatter(x=y_times_best, y=y_values_best,
                               mode='lines', line=dict(color='blue', width=2, dash='dash'),
                               name=f"better current aligned -> {best_key}", visible='legendonly'))
    traces_y.append(go.Scatter(x=y_times_s, y=y_values_s,
                               mode='lines', line=dict(color='green', width=2, dash='dash'),
                               name=f"better current aligned -> {s_key}", visible='legendonly'))
    traces_y.append(go.Scatter(x=scaled_y_times_best, y=scaled_y_values_best,
                               mode='lines', line=dict(color='blue', width=1, dash='dot'), 
                               name=f"current aligned+scaled -> {best_key}", visible='legendonly'))
    traces_y.append(go.Scatter(x=scaled_y_times_s, y=scaled_y_values_s,
                               mode='lines', line=dict(color='green', width=1, dash='dot'),
                               name=f"current aligned+scaled -> {s_key}", visible='legendonly'))

    # set graphing padding
    def padding_amount(a_min, a_max):
        span = a_max - a_min
        if span == 0:
            p = 0.05
        else:
            p = span * 0.05
        return a_min - p, a_max + p
    xmin = min([cur_x.min(), rx_b.min(), rx_s.min(), x_values_best.min(), x_values_s.min()])
    xmax = max([cur_x.max(), rx_b.max(), rx_s.max(), x_values_best.max(), x_values_s.max()])
    ymin = min([cur_y.min(), ry_b.min(), ry_s.min(), y_values_best.min(), y_values_s.min()])
    ymax = max([cur_y.max(), ry_b.max(), ry_s.max(), y_values_best.max(), y_values_s.max()])
    txmin = min([cur_times.min(), rt_b.min(), rt_s.min(), x_times_best.min(), x_times_s.min()])
    txmax = max([cur_times.max(), rt_b.max(), rt_s.max(), x_times_best.max(), x_times_s.max()])
    tymin = min([cur_times.min(), rt_b.min(), rt_s.min(), y_times_best.min(), y_times_s.min()])
    tymax = max([cur_times.max(), rt_b.max(), rt_s.max(), y_times_best.max(), y_times_s.max()])

    xr = padding_amount(xmin, xmax)
    yr = padding_amount(ymin, ymax)
    txr = padding_amount(txmin, txmax)
    tyr = padding_amount(tymin, tymax)

    # set axis ranges using previously computed tr, xr, yr (from 3D section)
    layout_x = go.Layout(title=f"X(t) comparison (current + {best_key} + {s_key})",
                         xaxis=dict(title="Time (s)", range=[max(0, txr[0]), txr[1]]),
                         yaxis=dict(title="X (normalized)", range=[xr[0], xr[1]]))
    layout_y = go.Layout(title=f"Y(t) comparison (current + {best_key} + {s_key})",
                         xaxis=dict(title="Time (s)", range=[max(0, tyr[0]), tyr[1]]),
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