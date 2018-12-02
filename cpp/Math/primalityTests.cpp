#include<bits/stdc++.h>
using namespace std;

#define REP(i, n) for(int i = 0; i < n; i++)

typedef long long Long;

bool isPrime(Long n) {
	if (n == 1) {
		return false;
	}
	if (n == 2) {
		return true;
	}
	if (n % 2 == 0) {
		return false;
	}
	for (int d = 3; (Long)d * d <= n; d += 2) {
		if (n % d == 0) {
			return false;
		}
	}
	return true;
}

void test(Long num) {
	cout << num << " " << isPrime(num) << '\n';
}
int main () {
	REP (i, 100) {
		test(i + 1);
	}
	test(1000000000000000);
	test((int)1e9 + 7);
}
