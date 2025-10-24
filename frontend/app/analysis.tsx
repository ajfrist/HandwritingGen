import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { API_URL } from '@/constants/api';
import React, { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, LayoutChangeEvent, PanResponder, ScrollView, StyleSheet, TextInput, TouchableOpacity, View } from 'react-native';
import Svg, { Path } from 'react-native-svg';

interface TouchPoint {
  x: number;
  y: number;
  x_norm: number;
  y_norm: number;
  timestamp: number;
  stroke_count: number;
}

export default function HandwritingAnalysisScreen() {
  // strokes: array of strokes; each stroke is an array of touch points
  const [strokes, setStrokes] = useState<TouchPoint[][]>([]);
  const canvasRef = useRef<View | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [resultText, setResultText] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [currentStrokeCount, setCurrentStrokeCount] = useState(0);
  const [currentUser, setCurrentUser] = useState<string>('');

  // Load current user on mount
  useEffect(() => {
    AsyncStorage.getItem('currentUser').then(user => {
      if (user) setCurrentUser(user);
    });
  }, []);

  const resetCanvas = () => {
    setStrokes([]);
    setCurrentStrokeCount(0);
    setResultText('');
  };

  const saveCharacter = async () => {
    if (strokes.length === 0 || !currentUser || !resultText) return;
    
    setIsSaving(true);
  const url = API_URL + '/save_user_character';
    
    try {
  const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          points_data: strokes.flat(),
          user: currentUser,
          ascii_char: resultText.split(/[\s()%]/)[0] // Get just the character, removing confidence and any whitespace
        })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setResultText(prev => prev + '\nSaved successfully!');
    } catch (err) {
      setResultText(prev => prev + '\nFailed to save: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setIsSaving(false);
    }
  };
  
  const createPoint = (x: number, y: number): TouchPoint => ({
    x,
    y,
    x_norm: canvasSize.width ? x / canvasSize.width : 0,
    y_norm: canvasSize.height ? y / canvasSize.height : 0,
    // store timestamps in seconds
    timestamp: Date.now() / 1000,
    stroke_count: currentStrokeCount
  });

  const panResponder = PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,
    onPanResponderGrant: (evt) => {
      const { locationX, locationY } = evt.nativeEvent;
      const newPoint = createPoint(locationX, locationY);
      // start a new stroke
      setStrokes(prev => [...prev, [newPoint]]);
    },
    onPanResponderMove: (evt) => {
      const { locationX, locationY } = evt.nativeEvent;
      const newPoint = createPoint(locationX, locationY);
      // append to current stroke
      setStrokes(prev => {
        if (prev.length === 0) return [[newPoint]];
        const copy = prev.map(s => s.slice());
        copy[copy.length - 1].push(newPoint);
        return copy;
      });
    },
    onPanResponderRelease: () => {
      // increment stroke count for next stroke
      setCurrentStrokeCount(prev => prev + 1);
    },
    onPanResponderTerminate: () => {
      // treat similar to release
      setCurrentStrokeCount(prev => prev + 1);
    },
  });

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1 }} keyboardShouldPersistTaps="handled">
      <ThemedView style={styles.container}>
        <ThemedText style={styles.title}>Handwriting Analysis</ThemedText>
        <ThemedText style={styles.instruction}>
          Write the following sentence:
        </ThemedText>
        <ThemedText style={styles.sampleText}>
          "The brown dog jumped over the lazy dog."
        </ThemedText>

        <View
          ref={r => { canvasRef.current = r; }}
          style={styles.canvas}
          onLayout={(e: LayoutChangeEvent) => {
            const { width, height } = e.nativeEvent.layout;
            setCanvasSize({ width, height });
          }}
          {...panResponder.panHandlers}
        >
          {/* Draw strokes as SVG paths */}
          <Svg width="100%" height="100%">
            {strokes.map((stroke, sIdx) => {
              if (!stroke || stroke.length === 0) return null;
              // build path: M x y L x y ...
              const d = stroke.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`).join(' ');
              return (
                <Path
                  key={`stroke-${sIdx}`}
                  d={d}
                  stroke="#000"
                  strokeWidth={3}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
              );
            })}
          </Svg>
        </View>

        <ThemedText style={styles.pointCount}>
          Strokes recorded: {strokes.length} • Points total: {strokes.reduce((acc, s) => acc + s.length, 0)}
        </ThemedText>

        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[styles.analyzeButton, isAnalyzing && styles.analyzeButtonDisabled]}
            onPress={async () => {
              if (strokes.length === 0 || isAnalyzing) return;
              setIsAnalyzing(true);
              setResultText('');

              const url = API_URL + '/analyze';
              
              try {
                const response = await fetch(url, {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json'
                  },
                  body: JSON.stringify({
                    points: strokes.flat()
                  })
                });

                if (!response.ok) {
                  throw new Error('Network response was not ok');
                }

                const data = await response.json();
                // Extract top-confidence character from returned list of dicts
                const extractTopChar = (payload: any): string | null => {
                  // payload may be an array of items or an object containing the array
                  let list: any[] | null = null;
                  if (Array.isArray(payload)) list = payload;
                  else if (payload && Array.isArray(payload.result)) list = payload.results;
                  else if (payload && Array.isArray(payload.letters)) list = payload.letters;

                  list = payload.results;
                  
                  if (!list) return null;

                  let bestChar: string | null = null;
                  let bestConf = -Infinity;
                  for (const item of list) {
                    if (!item) continue;
                    // item may be a dict like { char, confidence, sub_char }
                    const char = item.char ?? item[0] ?? item.ascii ?? item;
                    const conf = Number(item.confidence ?? item[1] ?? item.conf ?? item.score ?? NaN);
                    if (typeof char === 'string' && Number.isFinite(conf) && conf > bestConf) {
                      bestConf = conf;
                      bestChar = char;
                    }
                  }
                  if (bestChar !== null) {
                    // show as letter with confidence percentage if available
                    const pct = Number.isFinite(bestConf) ? Math.round(bestConf * 100) : null;
                    return pct !== null ? `${bestChar} (${pct}%)` : bestChar;
                  }
                  return null;
                };

                const top = extractTopChar(data);
                if (top) setResultText(top);
                else setResultText("Unknown");
                // else setResultText(typeof data === 'string' ? data : JSON.stringify(data));
              } catch (err) {
                setResultText('Failed to analyze: ' + (err instanceof Error ? err.message : String(err)));
              } finally {
                setIsAnalyzing(false);
              }
            }}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? <ActivityIndicator color="#fff" /> : <ThemedText style={styles.analyzeButtonText}>Analyze</ThemedText>}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.resetButton]}
            onPress={resetCanvas}
          >
            <ThemedText style={styles.buttonText}>Reset</ThemedText>
          </TouchableOpacity>
        </View>

        <View style={styles.resultContainer}>
          <TextInput
            style={[styles.resultInput, { borderColor: '#666' }]}
            value={resultText}
            onChangeText={setResultText}
            placeholder="Analysis result will appear here. You can edit this text if needed."
            placeholderTextColor="#999"
            multiline
            editable={true}
          />
          <TouchableOpacity
            style={[styles.saveButton, (!strokes.length || !resultText || !currentUser) && styles.saveButtonDisabled]}
            onPress={saveCharacter}
            disabled={!strokes.length || !resultText || !currentUser || isSaving}
          >
            {isSaving ? <ActivityIndicator color="#fff" /> : <ThemedText style={styles.buttonText}>Save</ThemedText>}
          </TouchableOpacity>
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
  sampleText: {
    fontSize: 16,
    fontStyle: 'italic',
    marginBottom: 20,
    color: '#000000',
  },
  canvas: {
    flex: 1,
    backgroundColor: '#f0f0f0',
    borderWidth: 1,
    borderColor: '#cccccc',
    borderRadius: 5,
    position: 'relative',
  },
  touchPoint: {
    position: 'absolute',
    width: 4,
    height: 4,
    backgroundColor: '#000000',
    borderRadius: 2,
  },
  pointCount: {
    marginTop: 10,
    fontSize: 14,
    color: '#666666',
  },
  analyzeButton: {
    flex: 1,
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    height: 44,
    justifyContent: 'center',
  },
  analyzeButtonDisabled: {
    backgroundColor: '#999999',
  },
  analyzeButtonText: {
    color: '#ffffff',
    fontWeight: 'bold',
  },
  resultInput: {
    flex: 1,
    minHeight: 80,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 6,
    padding: 10,
    color: '#000',
    backgroundColor: '#fff',
  },
  buttonRow: {
    flexDirection: 'row',
    marginTop: 16,
    gap: 10,
  },
  resetButton: {
    flex: 1,
    backgroundColor: '#FF3B30',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    height: 44,
    justifyContent: 'center',
  },
  resultContainer: {
    flexDirection: 'row',
    marginTop: 12,
    gap: 10,
  },
  saveButton: {
    backgroundColor: '#34C759',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 80,
  },
  saveButtonDisabled: {
    backgroundColor: '#999999',
  },
  buttonText: {
    color: '#ffffff',
    fontWeight: 'bold',
  },
});