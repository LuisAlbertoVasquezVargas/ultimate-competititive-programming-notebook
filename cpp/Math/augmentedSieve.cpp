
struct Math {
  vInt smallestFactor;
  Math() {}
  Math(int max_n) { 
    sieve(max_n); 
  }

  void sieve(int max_n) {
    smallestFactor = vInt(max_n + 1, -1);  // TODO(luisvasquez): to change this line to something more readable.
    smallestFactor[0] = smallestFactor[1] = 0;
    for (int i = 2; i * i <= max_n; ++i) {
      if (smallestFactor[i] == -1) {
        for (int j = i * i; j <= max_n; j += i) {
          smallestFactor[j] = i;
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
