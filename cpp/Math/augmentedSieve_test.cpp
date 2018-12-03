typedef long long Long;
typedef vector<int> vInt;
typedef vector<vInt> vvInt;
typedef vector<vvInt> vvvInt;
typedef vector<vvvInt> vvvvInt;
typedef vector<string> vStr;
typedef unordered_map<int, int> hashMap;
typedef set<int> hashSet;
typedef pair<int, int> Pair;
typedef vector<Pair> vPair;

#define REP(i, n) for (int i = 0; i < n; i++)
#define ALL(v) v.begin(), v.end()
#define SZ(v) (int)v.size()

struct Math {
  vInt smallestFactor;
  Math() {}
  Math(int max_n) { 
    sieve(max_n); 
  }

  void sieve(int max_n) {
    smallestFactor = vInt(max_n + 1, -1);  // TODO(luisvasquez): to rename p.
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

class Solution {
  vInt P;
  vInt pa;
  const int N = (int)1e5;
 public:
  int largestComponentSize(vector<int> &A) {  
    Math math(N);
    int n = SZ(A);
    vvInt buckets(n);
    vInt primes;
    REP(i, n) {
      vInt p, e;
      math.primefact(A[i], p, e);
      buckets[i] = p;
      primes.insert(primes.end(), ALL(p));
    }
    sort(ALL(primes));
    unique(ALL(primes)) - primes.begin();
    hashMap getPos;
    int it = 0;
    for (int prime : primes) {
      getPos[prime] = it++;
    }

    initUF(n);
    vvInt indices(SZ(primes));
    REP(i, n) {
      for (auto p : buckets[i]) {
        int pos = getPos[p];
        indices[pos].push_back(i);
      }
    }

    for (vInt index : indices) {
      REP(i, SZ(index) - 1) { Union(index[i], index[i + 1]); }
    }

    hashMap freq;
    int ans = 0;
    REP(i, n) {
      freq[Find(i)]++;
      ans = max(ans, freq[Find(i)]);
    }
    return ans;
  }

  void initUF(int n) {
    pa = vInt(n);
    REP(i, n) { pa[i] = i; }
  }

  int Find(int x) { return pa[x] = (pa[x] == x ? x : Find(pa[x])); }
  void Union(int x, int y) { pa[Find(x)] = Find(y); }
};
