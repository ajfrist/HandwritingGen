import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Image, StyleSheet } from 'react-native';

export default function HomeScreen() {
  return (
    <ThemedView style={styles.container}>
      <ThemedText style={styles.title}>Welcome to HandwritingTracker</ThemedText>
      <ThemedView style={styles.imageContainer}>
        <Image source={require('@/assets/images/pen.png')} style={styles.headerImage} resizeMode="contain" />
      </ThemedView>
      <ThemedText style={styles.description}>
        This application allows you to train and synthesize handwriting. Use the menu to:
      </ThemedText>
      <ThemedText style={styles.bullet}>
        • Train: Add new users and manage training data
      </ThemedText>
      <ThemedText style={styles.bullet}>
        • Synthesize: Generate handwritten text based on trained data
      </ThemedText>
      <ThemedText style={styles.instruction}>
        Get started by selecting a user in the Train section or creating a new one.
      </ThemedText>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    alignItems: 'flex-start',
    justifyContent: 'flex-start',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  description: {
    fontSize: 16,
    marginBottom: 15,
  },
  bullet: {
    fontSize: 16,
    marginLeft: 10,
    marginBottom: 10,
  },
  instruction: {
    fontSize: 16,
    marginTop: 20,
    fontStyle: 'italic',
  },
  headerImage: {
    width: 100,
    height: 100,
  },
  imageContainer: {
    width: 140,
    height: 140,
    backgroundColor: '#ffffff',
    borderRadius: 70,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    marginBottom: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
});