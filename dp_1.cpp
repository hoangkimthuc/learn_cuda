// dp_5.cpp
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

int myrand(int min, int max) {
  return rand()%(max-min)+min;
}

void lock(std::atomic<int>& m) {
  while (m)
    ; // busy waiting
  m=1;
}

void unlock(std::atomic<int>& m) {
  m=0;
}

void phil(int ph, std::atomic<int>& ma, std::atomic<int>& mb) {
  
  int duration=myrand(1000, 2000);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));
  std::cout<<ph<<" thinks "<<duration<<"ms\n";

  lock(ma);
  std::cout<<"\t\t"<<ph<<" got fork " << " ma " <<"\n";
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));


  lock(mb);
  std::cout<<"\t\t"<<ph<<" got fork " << " mb " <<"\n";
  
  duration=myrand(1000, 2000);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));
  std::cout<<"\t\t\t\t"<<ph<<" eats "<<duration<<"ms\n";

  unlock(mb);
  unlock(ma);
  
}

int main() {
  std::cout<<"dp_5\n";
  srand(time(nullptr));

  std::atomic<int> m1{0}, m2{0}, m3{0}, m4{0};

  std::thread t1([&] {phil(1, m1, m2);});
  std::thread t2([&] {phil(2, m2, m3);});
  std::thread t3([&] {phil(3, m3, m4);});
  std::thread t4([&] {phil(4, m1, m1);});

  t1.join();
  t2.join();
  t3.join();
  t4.join();
}