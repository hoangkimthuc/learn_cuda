#include <iostream>
#include <atomic>
#include <thread>

std::atomic<int> atomic_flag = ATOMIC_VAR_INIT(0);

void thread1() {
    // set the flag to 1
    atomic_flag.store(1);
    // perform an atomic memory fence
    std::atomic_thread_fence(std::memory_order_release);
}

void thread2() {
    // perform an atomic memory fence
    std::atomic_thread_fence(std::memory_order_acquire);
    // check the flag
    int flag_value = atomic_flag.load();
    std::cout << "Flag value: " << flag_value << std::endl;
}

int main() {
    // create two threads and run them concurrently
    std::thread t1(thread1);
    std::thread t2(thread2);

    t1.join();    // wait for t1 to complete
    t2.join();    // wait for t2 to complete

    return 0;
}