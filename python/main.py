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

from data_structures import TouchPoint, Stroke, Character;
from visualization import visualize_strokes, visualize_parametric, visualize_comparison
from character_matching import identify_character, identify_screen_characters, REFERENCE_CHARACTERS
from character_matching import normalize_positions, trim_leading_time


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
                print("\nResetting...\n")
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
                print("Compiled Identification results:")
                for item in results:
                    # item expected as (char, confidence, sub_char)
                    try:
                        ch, conf, sub = item
                        print(f"  {ch}: {conf:.2f}")
                    except Exception as e:
                        print("  ", item)
                        raise e
                for item in results:
                    ch, conf, sub = item
                    if (conf > 0.7):
                        print(f"Character '{ch}' identified with confidence {conf:.2f}")
                print()

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
                    raise e;

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
                print("Stroke ended")

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
