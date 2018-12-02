#include <bits/stdc++.h>
using namespace std;

typedef long long Long;

Long fastPow(Long a, Long b, Long c) {
  Long ans = 1;
  while (b > 0) {
    if (b & 1) {
      ans = (ans * a) % c;
    }
    b >>= 1;
    a = (a * a) % c;
  }
  return ans;
}

int main() {
	
}
