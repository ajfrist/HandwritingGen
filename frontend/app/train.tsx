import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Picker } from '@react-native-picker/picker';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Modal, ScrollView, StyleSheet, TextInput, TouchableOpacity, View } from 'react-native';

interface User {
  id: string;
  name: string;
  lettersCount: number;
}

export default function TrainScreen() {
  const [users, setUsers] = useState<User[]>([]);
  const [currentUser, setCurrentUser] = useState<string>('');
  const [newUserName, setNewUserName] = useState<string>('');
  const [showNewUserInput, setShowNewUserInput] = useState(false);
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [showOptions, setShowOptions] = useState(false);
  const router = useRouter();

  useEffect(() => {
    loadUsers();
    loadCurrentUser();
  }, []);

  const loadUsers = async () => {
    try {
      const usersData = await AsyncStorage.getItem('users');
      if (usersData) {
        setUsers(JSON.parse(usersData));
      }
    } catch (error) {
      console.error('Error loading users:', error);
    }
  };

  const loadCurrentUser = async () => {
    try {
      const current = await AsyncStorage.getItem('currentUser');
      if (current) {
        setCurrentUser(current);
      }
    } catch (error) {
      console.error('Error loading current user:', error);
    }
  };

  const saveCurrentUser = async (userId: string) => {
    try {
      await AsyncStorage.setItem('currentUser', userId);
      setCurrentUser(userId);
    } catch (error) {
      console.error('Error saving current user:', error);
    }
  };

  const createNewUser = async () => {
    if (!newUserName.trim()) return;

    const newUser: User = {
      id: Date.now().toString(),
      name: newUserName.trim(),
      lettersCount: 0,
    };

    try {
      const updatedUsers = [...users, newUser];
      await AsyncStorage.setItem('users', JSON.stringify(updatedUsers));
      setUsers(updatedUsers);
      setNewUserName('');
      setShowNewUserInput(false);
      saveCurrentUser(newUser.id);
    } catch (error) {
      console.error('Error creating new user:', error);
    }
  };

    const handleRemoveUser = async (userId: string) => {
    try {
        const updatedUsers = users.filter(user => user.id !== userId);
        await AsyncStorage.setItem('users', JSON.stringify(updatedUsers));
        setUsers(updatedUsers);
        if (currentUser === userId) {
        await AsyncStorage.removeItem('currentUser');
        setCurrentUser('');
        }
    } catch (error) {
        console.error('Error removing user:', error);
    }
    setShowOptions(false);
  };

const handleTrainUser = (userId: string) => {
  setShowOptions(false);
  router.push({
    pathname: '/analysis',
    params: { userId }
  });
};

  return (
    <ThemedView style={styles.container}>
      <ThemedText style={styles.title}>Training Data Management</ThemedText>
      
      <ThemedView style={styles.currentUserSection}>
        <ThemedText style={styles.label}>Current User:</ThemedText>
        <Picker
          selectedValue={currentUser}
          style={styles.picker}
          onValueChange={(itemValue) => saveCurrentUser(itemValue)}>
          <Picker.Item label="Select a user" value="" />
          {users.map((user) => (
            <Picker.Item key={user.id} label={user.name} value={user.id} />
          ))}
        </Picker>
      </ThemedView>

      {showNewUserInput ? (
        <ThemedView style={styles.newUserSection}>
          <TextInput
            style={styles.input}
            value={newUserName}
            onChangeText={setNewUserName}
            placeholder="Enter new user name"
            placeholderTextColor="#999"
          />
          <TouchableOpacity style={styles.button} onPress={createNewUser}>
            <ThemedText>Create User</ThemedText>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.cancelButton}
            onPress={() => {
              setShowNewUserInput(false);
              setNewUserName('');
            }}>
            <ThemedText>Cancel</ThemedText>
          </TouchableOpacity>
        </ThemedView>
      ) : (
        <TouchableOpacity
          style={styles.button}
          onPress={() => setShowNewUserInput(true)}>
          <ThemedText>Add New User</ThemedText>
        </TouchableOpacity>
      )}

      <ScrollView style={styles.userList}>
        <ThemedText style={styles.subtitle}>User List</ThemedText>
        {users.map((user) => (
          <ThemedView key={user.id} style={styles.userItem}>
            <View style={styles.userInfo}>
              <ThemedText style={styles.userName}>{user.name}</ThemedText>
              <ThemedText style={styles.userStats}>
                Letters trained: {user.lettersCount}
              </ThemedText>
            </View>
            <TouchableOpacity
              onPress={() => {
                setSelectedUser(user.id);
                setShowOptions(true);
              }}
              style={styles.optionsButton}
            >
              <Ionicons name="ellipsis-vertical" size={24} color="#000" />
            </TouchableOpacity>
          </ThemedView>
        ))}
      </ScrollView>

      <Modal
        visible={showOptions}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowOptions(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          onPress={() => setShowOptions(false)}
        >
          <View style={styles.optionsContainer}>
            <TouchableOpacity
              style={styles.optionItem}
              onPress={() => selectedUser && handleTrainUser(selectedUser)}
            >
              <ThemedText style={styles.optionText}>Train</ThemedText>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.optionItem, styles.removeOption]}
              onPress={() => selectedUser && handleRemoveUser(selectedUser)}
            >
              <ThemedText style={[styles.optionText, styles.removeText]}>
                Remove
              </ThemedText>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      </Modal>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  currentUserSection: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    marginBottom: 5,
  },
  picker: {
    height: 50,
    width: '100%',
  },
  newUserSection: {
    marginBottom: 20,
  },
  input: {
    height: 40,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 10,
    marginBottom: 10,
    color: '#000',
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 5,
    alignItems: 'center',
    marginBottom: 10,
  },
  cancelButton: {
    backgroundColor: '#FF3B30',
    padding: 10,
    borderRadius: 5,
    alignItems: 'center',
  },
  userList: {
    flex: 1,
  },
  subtitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  userItem: {
    padding: 15,
    borderRadius: 5,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#ccc',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  userInfo: {
    flex: 1,
  },
  userName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  userStats: {
    fontSize: 14,
  },
  optionsButton: {
    padding: 10,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  optionsContainer: {
    backgroundColor: '#ffffff',
    borderRadius: 10,
    padding: 10,
    width: '80%',
    maxWidth: 300,
  },
  optionItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  optionText: {
    fontSize: 16,
    color: '#007AFF',
  },
  removeOption: {
    borderBottomWidth: 0,
  },
  removeText: {
    color: '#FF3B30',
  },
});