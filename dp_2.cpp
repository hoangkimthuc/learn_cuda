#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;

mutex chopsticks[5];

void philosopher(int id) {
    
    while (true) {
      
    // philosopher is thinking
    cout << "Philosopher " << id << " is thinking." << endl;
    // this_thread::sleep_for(chrono::milliseconds(100));

    // philosopher tries to get both chopsticks/forks
    chopsticks[id].lock();    
    cout << "Philosopher " << id << " got fork " << id << endl;

    chopsticks[(id + 1) % 5].lock();
    cout << "Philosopher " << id << " got fork " << (id + 1) % 5 << endl;

    // philosopher is eating
    cout << "Philosopher " << id << " is eating." << endl;
    // this_thread::sleep_for(chrono::milliseconds(200));

    // philosopher releases both chopsticks/forks
    chopsticks[id].unlock();
    chopsticks[(id + 1) % 5].unlock();

    }
    
}

int main() {
    thread philosophers[5];

    for (int i = 0; i < 5; i++) {
        philosophers[i] = thread(philosopher, i);
    }

    for (int i = 0; i < 5; i++) {
        philosophers[i].join();
    }

    return 0;
}