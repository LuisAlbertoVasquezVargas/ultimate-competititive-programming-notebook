#include<bits/stdc++.h>
using namespace std;

typedef long long Long;
typedef vector<int> vInt;

#define PB push_back

// TODO(luisvasquez): to use bitsets
const int MAX_N = 1e6;
bool prime[MAX_N + 1];
vInt primes;

void init() {
    memset(prime, true, sizeof(prime));
    prime[0] = prime[1] = 0;
	for (int i = 2; i * i <= MAX_N; ++i) {
        if (prime[i]) {
            for (int j = i * i; j <= MAX_N ;j += i) {
                prime[j] = false;
            }
        }
    }
    for (int i = 2; i <= MAX_N; ++i) {
        if (prime[i]) {
            primes.PB(i);
        }
    }
}

bool isPrime(Long n) {
    for (int p:primes) {
        if ((Long)p * p > n) {
            break;
        }
        if (n % p == 0) {
            return false;
        }
    }
    return true;
}

// TODO(luisvasquez): to add main / testing