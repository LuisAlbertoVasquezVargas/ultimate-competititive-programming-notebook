#include<bits/stdc++.h>
using namespace std;

typedef long long Long;
typedef vector<int> vInt;
typedef vector<vInt> vvInt;
typedef vector<vvInt> vvvInt;
typedef vector<string> vStr;
typedef pair<int, int> Pair;
typedef vector<Pair> vPair;

#define REP(i,n) for(int i = 0; i < n; i++)
#define ALL(v) v.begin(), v.end()
#define SZ(v) (int)v.size()

struct Math {
  vInt smallestFactor;
  Math() {}
  Math(int max_n) { 
    sieve(max_n); 
  }

  void sieve(int max_n) {
    smallestFactor.assign(max_n + 1, -1);
    smallestFactor[0] = smallestFactor[1] = 0;
    for (int p = 2; p * p <= max_n; ++p) {
      if (smallestFactor[p] == -1) {
        for (int mult = p * p; mult <= max_n; mult += p) {
          smallestFactor[mult] = p;
        }
      }
    }
  }

  void primefact(int num, vInt &pr, vInt &ex) {
    while (true) {
      int p = smallestFactor[num];
      if (num == 1) {
        return;
      }

      if (p == -1) {
        pr.push_back(num);
        ex.push_back(1);
        break;
      }
      int exp = 0;
      while (num % p == 0) {
        exp++;
        num /= p;
      }
      pr.push_back(p);
      ex.push_back(exp);
    }
  }

  vInt getDivisors(vInt primes, vInt exps) {
    vInt divisors = {1};

    REP (index, SZ(primes)) {
      int prime = primes[index];
      int exp = exps[index];
      int curLen = SZ(divisors);
      REP (e, exp) {
        REP (ptr, curLen) {
          divisors.push_back(prime * divisors[ptr + e * curLen]);
        }
      }
    }
    return divisors;
  }
};

int main() {
  const int MAX_VAL = (int)1e6;
	Math math(MAX_VAL);
	int n;
	scanf("%d", &n);
  vInt AC(MAX_VAL + 1);
	REP (i, n) {
    int x;
    scanf("%d", &x);
    vInt p, e;
    math.primefact(x, p, e);
    vInt divisors = math.getDivisors(p, e);
    for (int divisor : divisors) {
      AC[divisor] ++;
    }
  }
  int ans = 0;
	REP (i, MAX_VAL + 1) {
    if (AC[i] >= 2) {
      ans = i;
    }
  }
	printf("%d\n", ans);
}