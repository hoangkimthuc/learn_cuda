#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx; // Declare a global mutex

int shared_variable = 0;

void increment_shared_variable(int num_increments) {
    for (int i = 0; i < num_increments; i++) {
        mtx.lock(); // Acquire the lock on the mutex
        shared_variable++;
        mtx.unlock(); // Release the lock on the mutex
    }
}

int main() {
    const int num_threads = 5;
    const int num_increments_per_thread = 100000;

    std::thread threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        threads[i] = std::thread(increment_shared_variable, num_increments_per_thread);
    }

    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }

    std::cout << "Shared variable value: " << shared_variable << std::endl;

    return 0;
}