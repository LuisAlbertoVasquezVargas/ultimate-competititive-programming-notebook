#include<bits/stdc++.h>
using namespace std;

#define REP(i, n) for(int i = 0; i < n; i++)

const int MAX_N = 100000;
bitset <MAX_N + 1> prime(1);

void sieve(){
	prime.set();
	prime[0] = false;
	prime[1] = false;
	for (int i = 2; i * i <= MAX_N; i++) {
		if (prime[i]) {
			for (int j = i * i; j <= MAX_N; j += i) {
				prime[j] = false;
			}
		}
	}
}

void test(int num) {
	cout << num << " " << prime.test(num) << '\n';
}

int main() {
	sieve();
	REP (i, 100) {
		test(i + 1);
	}
}


