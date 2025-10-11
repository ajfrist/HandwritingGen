import pygame
import time
import numpy as np
import os
import glob
import pickle
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
    
    def add_stroke(self, stroke):
        if len(stroke) > 0:  # Only add non-empty strokes
            self.strokes.append(stroke)
    
    def __len__(self):
        return len(self.strokes)
    
    def __iter__(self):
        return iter(self.strokes)
    
    def all_points(self):
        """Iterator over all points in all strokes"""
        for stroke in self.strokes:
            yield from stroke


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
# Main logic
# ------------------------

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

def reset_state():
    """Reset all state variables to initial values"""
    global current_character, current_stroke, recording, playback_start_time, recording_start_time
    current_character = Character()
    current_stroke = Stroke()
    recording = True
    playback_start_time = None
    recording_start_time = None

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
                save_character(current_character)
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
        for stroke in current_character:
            pts = [
                (p.get_x(WIDTH), p.get_y(HEIGHT))
                for p in stroke if p.timestamp <= elapsed
            ]
            if len(pts) > 1:
                pygame.draw.lines(screen, (200, 200, 0), False, pts, 2)

    pygame.display.flip()

pygame.quit()
raise SystemExit
