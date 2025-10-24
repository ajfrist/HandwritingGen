import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Picker } from '@react-native-picker/picker';

import { API_URL } from '@/constants/api';
import React, { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, TextInput, TouchableOpacity, View } from 'react-native';
import Svg, { Path } from 'react-native-svg';

interface Point {
  // server may return either normalized coords (x_norm/y_norm) or pixel coords (x/y)
  x_norm?: number;
  y_norm?: number;
  x?: number;
  y?: number;
  timestamp: number;
  stroke_count?: number;
}

export default function HandwritingSynthesisScreen() {
  const [inputText, setInputText] = useState('');
  const [points, setPoints] = useState<Point[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState('');
  const [generationResult, setGenerationResult] = useState<string>('');

  const [users, setUsers] = useState<{ id: string; name: string; }[]>([]);
  const [currentUser, setCurrentUser] = useState<string>('');

  // playback state
  const [playStrokes, setPlayStrokes] = useState<{ x: number; y: number; }[][]>([]);
  const playCancelRef = useRef<{ cancelled: boolean }>({ cancelled: false });

  const handleGenerate = async () => {
    if (!inputText.trim() || isGenerating) return;
    setIsGenerating(true);
    setError('');

    try {
  const url = API_URL + '/generate';
      // send user display name instead of id
      const userName = users.find(u => u.id === currentUser)?.name || currentUser;

  const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          // send selected user and the input text
          user: userName,
          text: inputText.trim(),
          points: [],  // Empty for generation request
          rect_left: canvasLayout.left,
          rect_top: canvasLayout.top,
          rect_width: canvasLayout.width,
          rect_height: canvasLayout.height
        })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // If server returns a list-of-lists containing [ascii, confidence, obj], extract top ascii by confidence
      let topAscii: string | null = null;
      
      if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
        try {
          let bestConf = -Infinity;
          for (const item of data) {
            if (!Array.isArray(item)) continue;
            const ascii = item[0];
            const conf = Number(item[1]);
            if (typeof ascii === 'string' && Number.isFinite(conf) && conf > bestConf) {
              bestConf = conf;
              topAscii = ascii;
            }
          }
        } catch (e) {
          // ignore parse errors
        }
      } else if (data && Array.isArray(data.letters)) {
        // alternative location: data.letters = [[ascii, confidence, obj], ...]
        try {
          let bestConf = -Infinity;
          for (const item of data.letters) {
            if (!Array.isArray(item)) continue;
            const ascii = item[0];
            const conf = Number(item[1]);
            if (typeof ascii === 'string' && Number.isFinite(conf) && conf > bestConf) {
              bestConf = conf;
              topAscii = ascii;
            }
          }
        } catch (e) {
          // ignore
        }
      }

      if (topAscii) setGenerationResult(topAscii);

      if (data.points) {
        const processed = processIncomingPoints(data.points);
        setPoints(processed);
        // start playback
        playCancelRef.current.cancelled = false;
        setPlayStrokes([]);
        playPoints(processed);
      } else if (!topAscii) {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      setError('Failed to generate: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setIsGenerating(false);
    }
  };



  useEffect(() => {
    loadUsers();
    loadCurrentUser();
    return () => {
      playCancelRef.current.cancelled = true;
    };
  }, []);

  const loadUsers = async () => {
    try {
      const usersData = await AsyncStorage.getItem('users');
      if (usersData) setUsers(JSON.parse(usersData));
    } catch (e) {
      console.error('loadUsers error', e);
    }
  };

  const loadCurrentUser = async () => {
    try {
      const cur = await AsyncStorage.getItem('currentUser');
      if (cur) setCurrentUser(cur);
    } catch (e) {
      console.error('loadCurrentUser', e);
    }
  };

  // Play points according to timestamps. Points can have timestamps in seconds (0-10) or ms (>1000).
  const processIncomingPoints = (rawPoints: any[]): Point[] => {
    if (!Array.isArray(rawPoints)) return [];

    // map original stroke_count values to sequential indices in order of first appearance
    const strokeMap = new Map<number, number>();
    let nextIdx = 0;

    const processed: Point[] = rawPoints.map((p: any) => {
      const rawStroke = (typeof p.stroke_count === 'number' && Number.isFinite(p.stroke_count)) ? Math.floor(p.stroke_count) : 0;
      if (!strokeMap.has(rawStroke)) {
        strokeMap.set(rawStroke, nextIdx++);
      }
      const mappedStroke = strokeMap.get(rawStroke) ?? 0;

      // determine normalized coords from either x_norm/y_norm or x/y (pixels relative to canvasLayout)
      let xNorm = 0;
      let yNorm = 0;
      if (typeof p.x_norm === 'number' && Number.isFinite(p.x_norm) && typeof p.y_norm === 'number' && Number.isFinite(p.y_norm)) {
        xNorm = p.x_norm;
        yNorm = p.y_norm;
      } else if (typeof p.x === 'number' && typeof p.y === 'number' && Number.isFinite(p.x) && Number.isFinite(p.y) && canvasLayout.width > 0 && canvasLayout.height > 0) {
        xNorm = (p.x - canvasLayout.left) / canvasLayout.width;
        yNorm = (p.y - canvasLayout.top) / canvasLayout.height;
      }
      // clamp
      xNorm = Math.max(0, Math.min(1, xNorm));
      yNorm = Math.max(0, Math.min(1, yNorm));

      const timestamp = (typeof p.timestamp === 'number' && Number.isFinite(p.timestamp)) ? p.timestamp : Number(p.timestamp) || 0;

      return { x_norm: xNorm, y_norm: yNorm, timestamp, stroke_count: mappedStroke } as Point;
    });

    return processed;
  };

  const playPoints = async (pts: Point[]) => {
    if (!pts || pts.length === 0) return;

    // ensure points sorted by timestamp
    const pointsSorted = [...pts].sort((a, b) => a.timestamp - b.timestamp);
    const timestamps = pointsSorted.map(p => p.timestamp);
    const maxTs = Math.max(...timestamps);
    const multiplier = maxTs < 10 ? 1000 : 1; // if timestamps look like seconds, convert to ms

    // reset
    setPlayStrokes([]);

    let prevTs = pointsSorted[0].timestamp;

    for (let i = 0; i < pointsSorted.length; i++) {
      if (playCancelRef.current.cancelled) break;
      const p = pointsSorted[i];
      const delay = Math.max(0, (p.timestamp - prevTs) * multiplier);
      // wait for the next point time
      // small safety clamp
      const waitMs = Math.min(delay, 5000);
      await new Promise(res => setTimeout(res, waitMs));

      if (playCancelRef.current.cancelled) break;

      setPlayStrokes(prev => {
        const copy = prev.map(s => s.slice());
        // defensive: ensure stroke_count is a valid non-negative integer
        let idx = typeof p.stroke_count === 'number' && Number.isFinite(p.stroke_count) ? Math.max(0, Math.floor(p.stroke_count)) : 0;
        // p should already be normalized by processIncomingPoints; but guard anyway
        let xNorm = (typeof p.x_norm === 'number' && Number.isFinite(p.x_norm)) ? p.x_norm : 0;
        let yNorm = (typeof p.y_norm === 'number' && Number.isFinite(p.y_norm)) ? p.y_norm : 0;
        // clamp to [0,1]
        xNorm = Math.max(0, Math.min(1, xNorm));
        yNorm = Math.max(0, Math.min(1, yNorm));

        while (copy.length <= idx) copy.push([]);
        copy[idx].push({ x: xNorm * canvasWidth, y: yNorm * canvasHeight });
        return copy;
      });

      prevTs = p.timestamp;
    }
  };

  // Convert normalized points to screen coordinates
  const canvasWidth = 300;  // Fixed width for demo
  const canvasHeight = 300; // Fixed height for demo

  const [canvasLayout, setCanvasLayout] = useState({ left: 0, top: 0, width: canvasWidth, height: canvasHeight });

  const renderPoints = (usePlayback = false) => {
    const src = usePlayback ? playStrokes : points.reduce((acc: { x:number;y:number;}[][], point) => {
        // defensive: stroke index
        let idx = typeof point.stroke_count === 'number' && Number.isFinite(point.stroke_count)
        ? Math.max(0, Math.floor(point.stroke_count))
        : 0;

        // determine normalized coords from either x_norm/y_norm or x/y (pixels relative to canvasLayout)
        let xNorm: number;
        let yNorm: number;
        if (typeof point.x_norm === 'number' && Number.isFinite(point.x_norm) &&
            typeof point.y_norm === 'number' && Number.isFinite(point.y_norm)) {
        xNorm = point.x_norm;
        yNorm = point.y_norm;
        } else if (typeof point.x === 'number' && typeof point.y === 'number' &&
                Number.isFinite(point.x) && Number.isFinite(point.y) &&
                canvasLayout.width > 0 && canvasLayout.height > 0) {
        // Convert pixel coordinates (absolute/relative to same coordinate system as canvasLayout)
        xNorm = (point.x - canvasLayout.left) / canvasLayout.width;
        yNorm = (point.y - canvasLayout.top) / canvasLayout.height;
        } else {
        xNorm = 0;
        yNorm = 0;
        }

        // clamp normalized coords to [0,1]
        xNorm = Math.max(0, Math.min(1, xNorm));
        yNorm = Math.max(0, Math.min(1, yNorm));

        // ensure the stroke array exists
        while (acc.length <= idx) acc.push([]);
        acc[idx].push({ x: xNorm * canvasWidth, y: yNorm * canvasHeight });
        return acc;
    }, [] as { x:number;y:number;}[][]);

    if (!src || src.length === 0) return null;

    return src.map((strokePoints, strokeIdx) => {
        if (!strokePoints || strokePoints.length === 0) return null;
        const d = strokePoints.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`).join(' ');
        return (
        <Path
            key={`stroke-${strokeIdx}`}
            d={d}
            stroke="#000"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
        />
        );
    });
  };

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1 }} keyboardShouldPersistTaps="handled">
      <ThemedView style={styles.container}>
        <ThemedText style={styles.title}>Handwriting Synthesis</ThemedText>
        <ThemedText style={styles.instruction}>
          Enter text to generate handwriting:
        </ThemedText>

        <ThemedText style={styles.label}>User:</ThemedText>
        <Picker
          selectedValue={currentUser}
          style={styles.picker}
          onValueChange={(val) => setCurrentUser(val)}
        >
          <Picker.Item label="Select a user" value="" />
          {users.map(u => <Picker.Item key={u.id} label={u.name} value={u.id} />)}
        </Picker>

        <TextInput
          style={styles.input}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Enter text..."
          placeholderTextColor="#999"
        />
        <TouchableOpacity
          style={[styles.generateButton, isGenerating && styles.generateButtonDisabled]}
          onPress={handleGenerate}
          disabled={isGenerating}
        >
          {isGenerating ? 
            <ActivityIndicator color="#fff" /> : 
            <ThemedText style={styles.generateButtonText}>Generate</ThemedText>
          }
        </TouchableOpacity>

        {error ? <ThemedText style={styles.error}>{error}</ThemedText> : null}
  {generationResult ? <ThemedText style={styles.resultText}>Top prediction: {generationResult}</ThemedText> : null}

        <View
          style={styles.canvas}
          onLayout={(e) => {
            const { x, y, width, height } = e.nativeEvent.layout;
            setCanvasLayout({ left: x, top: y, width, height });
          }}
        >
          <Svg width={canvasWidth} height={canvasHeight}>
            {/* render playback if available, otherwise render full points */}
            {playStrokes && playStrokes.length > 0 ? renderPoints(true) : renderPoints(false)}
          </Svg>
        </View>
      </ThemedView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#ffffff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#000000',
  },
  instruction: {
    fontSize: 18,
    marginBottom: 10,
    color: '#000000',
  },
  label: {
    fontSize: 16,
    marginBottom: 6,
    color: '#000000',
  },
  picker: {
    height: 50,
    width: '100%',
    marginBottom: 12,
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: '#cccccc',
    borderRadius: 5,
    paddingHorizontal: 10,
    marginBottom: 20,
    color: '#000000',
  },
  canvas: {
    width: 300,
    height: 300,
    backgroundColor: '#f0f0f0',
    borderWidth: 1,
    borderColor: '#cccccc',
    borderRadius: 5,
    marginTop: 20,
    alignSelf: 'center',
  },
  generateButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 5,
    alignItems: 'center',
    marginBottom: 20,
  },
  generateButtonDisabled: {
    backgroundColor: '#999999',
  },
  generateButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  error: {
    color: '#ff0000',
    marginBottom: 10,
  },

  resultText: {
    marginTop: 8,
    fontSize: 16,
    color: '#000',
  },
});