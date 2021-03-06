struct UnionFind {
  vInt parent;
  void initUF(int n) {
    parent.resize(n);
    iota(ALL(parent), 0);
  }

  int find(int x) { 
    return parent[x] = (parent[x] == x ? x : find(parent[x])); 
  }

  void unite(int x, int y) { 
    parent[find(x)] = find(y); 
  }
};

// TODO(luisvasquez) : review if it worths to use ranked unionFind. It needs more
// testing.

// TODO(luisvasquez) : instead of using unitUF to init the DS use a constructor.