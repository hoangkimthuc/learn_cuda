#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

using namespace std;

// Map function: takes a string and returns a vector of pairs
vector<pair<string, int>> mapper(const string& str) {
    vector<pair<string, int>> word_counts;
    string word = "";

    for (char c : str) {
        if (c == ' ') {
            word_counts.push_back(make_pair(word, 1));
            word = "";
        } else {
            word += c;
        }
    }

    word_counts.push_back(make_pair(word, 1));
    return word_counts;
}

// Reduce function: takes a key and a vector of values and returns the total count
int reducer(const string& key, const vector<int>& values) {
    int count = 0;
    for (int val : values) {
        count += val;
    }
    return count;
}

// MapReduce function: takes a vector of strings and outputs the word counts
vector<pair<string, int>> map_reduce(const vector<string>& inputs) {
    vector<thread> map_threads;
    vector<vector<pair<string, int>>> intermediate_results;

    // Map phase
    for (const string& input : inputs) {
        map_threads.push_back(thread([&intermediate_results](const string& input) {
            vector<pair<string, int>> results = mapper(input);
            intermediate_results.push_back(results);
        }, input));
    }

    for (thread& t : map_threads) {
        t.join();
    }

    // Shuffle phase
    vector<pair<string, vector<int>>> shuffled_results;
    for (vector<pair<string, int>>& intermediate_result : intermediate_results) {
        for (pair<string, int>& result : intermediate_result) {
            // Find the corresponding shuffled result
            auto it = find_if(shuffled_results.begin(), shuffled_results.end(), [result](const pair<string, vector<int>>& p) {
                return p.first == result.first;
            });
            if (it == shuffled_results.end()) {
                // If the key is not found, add a new pair
                shuffled_results.push_back(make_pair(result.first, vector<int>{result.second}));
            } else {
                // If the key is found, add the value to the existing vector
                it->second.push_back(result.second);
            }
        }
    }

    // Reduce phase
    vector<pair<string, int>> final_results;
    for (const pair<string, vector<int>>& shuffled_result : shuffled_results) {
        final_results.push_back(make_pair(shuffled_result.first, reducer(shuffled_result.first, shuffled_result.second)));
    }

    return final_results;
}

int main() {
    vector<string> inputs = {"hello world", "hello C++", "C++ is great", "world is great"};
    vector<pair<string, int>> results = map_reduce(inputs);

    for (const pair<string, int>& result : results) {
        cout << result.first << ": " << result.second << endl;
    }

    return 0;
}