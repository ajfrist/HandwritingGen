// constants/api.ts

import { Platform } from 'react-native';

let apiUrl = '';

if (typeof process !== 'undefined' && process.env && process.env.API_URL) {
  apiUrl = process.env.API_URL;
} else if (Platform.OS !== 'web') {
  // For Expo/React Native, use expo-constants if available
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const Constants = require('expo-constants').default;
    apiUrl = Constants?.expoConfig?.extra?.API_URL || '';
  } catch (e) {
    apiUrl = '';
  }
}

export const API_URL = apiUrl || 'http://10.110.29.137:5000';
