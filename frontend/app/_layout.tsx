import { Ionicons } from '@expo/vector-icons';
import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { Drawer } from 'expo-router/drawer';
import { StatusBar } from 'expo-status-bar';
import 'react-native-reanimated';

import { useColorScheme } from '@/hooks/use-color-scheme';

export default function RootLayout() {
  const colorScheme = useColorScheme();

  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Drawer
        screenOptions={{
          headerStyle: {
            backgroundColor: colorScheme === 'dark' ? '#000' : '#fff',
          },
          headerTintColor: colorScheme === 'dark' ? '#fff' : '#000',
          drawerStyle: {
            backgroundColor: colorScheme === 'dark' ? '#000' : '#fff',
          },
          drawerActiveTintColor: colorScheme === 'dark' ? '#fff' : '#000',
          drawerInactiveTintColor: colorScheme === 'dark' ? '#666' : '#999',
        }}>
        <Drawer.Screen
          name="index"
          options={{
            title: 'Home',
            drawerIcon: ({ color, size }) => (
              <Ionicons name="home-outline" size={size} color={color} />
            ),
          }}
        />
        <Drawer.Screen
          name="train"
          options={{
            title: 'Train',
            drawerIcon: ({ color, size }) => (
              <Ionicons name="pencil-outline" size={size} color={color} />
            ),
          }}
        />
        <Drawer.Screen
          name="analysis"
          options={{
            title: 'Analyze',
            drawerIcon: ({ color, size }) => (
              <Ionicons name="search" size={size} color={color} />
            ),
          }}
        />
        <Drawer.Screen
          name="synthesize"
          options={{
            title: 'Synthesize',
            drawerIcon: ({ color, size }) => (
              <Ionicons name="create-outline" size={size} color={color} />
            ),
          }}
        />
      </Drawer>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}
