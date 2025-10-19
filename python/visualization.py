import numpy as np

from interpolation import newton_interpolation, cubic_spline_interpolation
from character_matching import adjust_time_delay
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