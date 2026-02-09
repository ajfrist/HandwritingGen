import asyncio
import os
import glob
import pickle
import json
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from character_synthesis import (
    generate_sentence_character, 
    get_user_splines, 
    reset_loaded_user_splines,
    get_pixel_points
)
from data_structures import Character, Stroke, TouchPoint, load_reference_characters
from data_saving import save_character
from character_matching import identify_screen_characters


app = FastAPI(title="HandwritingGen API")

# Allow CORS from everywhere for development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def character_to_points(char: Character) -> list:
    """Convert a Character object to a list of point data."""
    points = []
    for stroke_idx, stroke in enumerate(char.strokes):
        for point in stroke:
            points.append({
                'x_norm': point.x_norm,
                'y_norm': point.y_norm,
                'timestamp': point.timestamp,
                'stroke_count': stroke_idx
            })
    return points


def points_to_character(points_data: list) -> Character:
    """Create a Character object from a list of point data."""
    char = Character()
    current_stroke = None
    current_stroke_count = -1
    
    for point in points_data:
        x = point.get('x_norm', 0.0)
        y = point.get('y_norm', 0.0)
        t = point.get('timestamp', 0.0)
        stroke_count = point.get('stroke_count', 0)
        
        if stroke_count != current_stroke_count:
            if current_stroke is not None and len(current_stroke) > 0:
                char.add_stroke(current_stroke)
            current_stroke = Stroke()
            current_stroke_count = stroke_count
        
        tp = TouchPoint(float(x), float(y), float(t))
        current_stroke.add_point(tp)
    
    if current_stroke is not None and len(current_stroke) > 0:
        char.add_stroke(current_stroke)
        
    return char


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/generate')
async def api_generate(payload: Dict[str, Any]):
    text = payload.get('text', '')
    user = payload.get('user', 'reference')
    points_data = payload.get('points', [])
    
    if not text and not points_data:
        print("asdfasdf")
        raise HTTPException(status_code=400, detail='text or points data required')

    # if points_data:
    #     # Create character from provided points
    #     char = Character()
    #     current_stroke = None
        
    #     for point in points_data:
    #         x = point.get('x_norm', 0.0)
    #         y = point.get('y_norm', 0.0)
    #         t = point.get('timestamp', 0.0)
    #         is_new_stroke = point.get('new_stroke', False)
            
    #         if is_new_stroke or current_stroke is None:
    #             if current_stroke is not None:
    #                 char.add_stroke(current_stroke)
    #             current_stroke = Stroke()
            
    #         tp = TouchPoint(float(x), float(y), float(t))
    #         current_stroke.add_point(tp)
        
    #     if current_stroke is not None and len(current_stroke) > 0:
    #         char.add_stroke(current_stroke)
    # else:

    # Get drawing rectangle dimensions from request
    rect_left = payload.get('rect_left', 0)
    rect_top = payload.get('rect_top', 0)
    rect_width = payload.get('rect_width', 800)  # Default to 800 if not provided
    rect_height = payload.get('rect_height', 800)  # Default to square if not provided

    # Generate character from text
    reset_loaded_user_splines()
    char = generate_sentence_character(text, user=user)

    # Convert to pixel coordinates for drawing
    points = get_pixel_points(char, rect_left, rect_top, rect_width, rect_height)
    print(f"Generated character for text '{text}' with {len(points)} points.")
    print(points)
    
    return {'points': points}


@app.get('/splines')
async def api_splines(user: str = 'reference'):
    reset_loaded_user_splines()
    splines = get_user_splines(user)
    keys = sorted(list(splines.keys()))
    return {'user': user, 'available': keys}


@app.get('/references')
async def api_references():
    refs = load_reference_characters()
    return {'refs': list(refs.keys())}


@app.post('/save_reference')
async def api_save_reference(payload: Dict[str, Any]):
    ascii_char = payload.get('ascii_char')
    points_data = payload.get('points', [])
    if not ascii_char or not points_data:
        raise HTTPException(status_code=400, detail='ascii_char and points data required')

    # Create character from points
    c = Character()
    current_stroke = None
    
    for point in points_data:
        x = point.get('x_norm', 0.0)
        y = point.get('y_norm', 0.0)
        t = point.get('timestamp', 0.0)
        is_new_stroke = point.get('new_stroke', False)
        
        if is_new_stroke or current_stroke is None:
            if current_stroke is not None:
                c.add_stroke(current_stroke)
            current_stroke = Stroke()
        
        tp = TouchPoint(float(x), float(y), float(t))
        current_stroke.add_point(tp)
    
    if current_stroke is not None and len(current_stroke) > 0:
        c.add_stroke(current_stroke)

    token = ascii_char
    if len(token) == 1 and token.isalpha():
        if token.isupper():
            token = token.lower() + 'c'
        else:
            token = token.lower()
    else:
        raise HTTPException(status_code=400, detail='ascii_char must be a single letter')

    ref_dir = os.path.join(os.path.dirname(__file__), 'character_references')
    os.makedirs(ref_dir, exist_ok=True)
    base = f'character_{token}'
    pattern = os.path.join(ref_dir, base + '*.pkl')
    existing = glob.glob(pattern)
    if not existing:
        path = os.path.join(ref_dir, base + '.pkl')
    else:
        i = 0
        while True:
            cand = os.path.join(ref_dir, f"{base}{i}.pkl")
            if not os.path.exists(cand):
                path = cand
                break
            i += 1

    try:
        with open(path, 'wb') as f:
            pickle.dump(c, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    reset_loaded_user_splines()
    return {'saved': os.path.basename(path)}


@app.post('/save_user_character')
async def api_save_user_character(payload: Dict[str, Any]):
    """Save a character to a user's personal collection."""
    points_data = payload.get('points_data', [])
    user = payload.get('user', 'default')
    ascii_char = payload.get('ascii_char')
    print(points_data)
    print(user)
    print(ascii_char)
    if not points_data or not ascii_char:
        raise HTTPException(status_code=400, detail='points data and ascii_char required')
        
    # Create character from points
    char = points_to_character(points_data)
    char.ascii_char = ascii_char
    
    # Use the save_character function from main.py
    try:
        saved_path = save_character(char, user=user, char_letter=ascii_char)
        if not saved_path:
            raise HTTPException(status_code=500, detail='Failed to save character')
        return {'saved': os.path.basename(saved_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/analyze')
async def api_analyze_points(payload: Dict[str, Any]):
    """Identify character(s) from an input list of points.

    Expects JSON: { "points": [ {x_norm, y_norm, timestamp, stroke_count}, ... ] }
    Returns: { "results": [ {"char": ..., "confidence": ..., "sub_char": ...}, ... ] }
    """
    points_data = payload.get('points', [])
    if not points_data:
        raise HTTPException(status_code=400, detail='points data required')

    # Build Character from points
    char = points_to_character(points_data)

    try:
        results = identify_screen_characters(char, threshold=0.3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    out = []
    for item in results:
        try:
            ch, conf, sub = item
        except Exception:
            # fallback if identify returns different shape
            ch = item
            conf = None
            sub = None
        out.append({'char': ch, 'confidence': float(conf) if conf is not None else None, 'sub_char': sub})

    return {'results': out}


# @app.post('/update_references')
# async def api_update_references():
#     ref_dir = os.path.join(os.path.dirname(__file__), 'character_references')
#     pkl_files = glob.glob(os.path.join(ref_dir, 'character_*.pkl'))
#     updated = []
#     skipped = []
#     errors = []
#     for pkl_file in pkl_files:
#         try:
#             filename = os.path.basename(pkl_file)
#             token = filename[len('character_'):].split('.')[0]
#             if len(token) >= 1 and token[0].isalpha():
#                 if len(token) == 2 and token[1].lower() == 'c':
#                     char_match = token[0].upper()
#                 else:
#                     char_match = token[0].lower()
#             else:
#                 skipped.append(filename)
#                 continue

#             with open(pkl_file, 'rb') as f:
#                 obj = pickle.load(f)
#             if not hasattr(obj, 'ascii_char') or obj.ascii_char is None or obj.ascii_char != char_match:
#                 obj.ascii_char = char_match
#                 with open(pkl_file, 'wb') as f:
#                     pickle.dump(obj, f)
#                 updated.append(filename)
#             else:
#                 skipped.append(filename)
#         except Exception as e:
#             errors.append({'file': pkl_file, 'error': str(e)})

#     reset_loaded_user_splines()
#     return {'updated': updated, 'skipped': skipped, 'errors': errors}


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    def get(self, client_id: str):
        return self.active_connections.get(client_id)


manager = ConnectionManager()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('python.api_server:app', host='0.0.0.0', port=5000, reload=True)
