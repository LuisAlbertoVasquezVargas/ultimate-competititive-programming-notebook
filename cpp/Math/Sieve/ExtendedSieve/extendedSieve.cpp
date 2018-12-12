// TODO(luisvasquez): to test again due changes.
// TODO(luisvasquez): to make more neutral code, no REP, no vInt.
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
    while (num != 1) {
      int p = smallestFactor[num];
      if (p == -1) {
        pr.push_back(num);
        ex.push_back(1);
        break;
      } else {
        int exp = 0;
        while (num % p == 0) {
          exp++;
          num /= p;
        }
        pr.push_back(p);
        ex.push_back(exp);
      }
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
