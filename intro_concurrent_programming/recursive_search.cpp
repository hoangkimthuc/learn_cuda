#include <iostream>
using namespace std;

int recursive_search(int arr[], int start, int end, int target) {
    if (start > end) {
    cout << "Index not found" << endl;
        return -1;
    }

    int mid = (start + end) / 2;
    if (arr[mid] == target) {
        return mid;
    } else if (arr[mid] < target) {
        return recursive_search(arr, mid + 1, end, target);
    } else {
        return recursive_search(arr, start, mid - 1, target);
    }
}
int main() {
    int int_arr[] = {1, 2, 3, 4, 5, 6};
    int target_int = 2;
    int array_len = sizeof(int_arr) / sizeof(int);
    int found_idx = recursive_search(int_arr, 0, array_len, target_int);
    cout << "Found index: " << found_idx << endl;
    return 0;
}