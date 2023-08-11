#include <iostream>
using namespace std;

void bubbleSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int myArray[] = {5, 2, 4, 6, 1, 3};
    int length = sizeof(myArray) / sizeof(myArray[0]);

    cout << "before sorting:" << endl;
    for (int i = 0; i < length; i++) {
        std::cout << myArray[i] << " ";
    }

    cout << "\n after sorting:" << endl;
    bubbleSort(myArray, length);

    for (int i = 0; i < length; i++) {
        std::cout << myArray[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
