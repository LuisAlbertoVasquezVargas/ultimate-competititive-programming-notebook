struct UnionFind {
	vInt parent;
  void initUF(int n) {
  	parent = vInt(n);
  	REP (i, n) {
  		parent[i] = i;
  	}
  }
  
  int find (int x) { 
		return parent[x] = (parent[x] == x ? x : Find(parent[x]));
  }
  
	void unite(int x, int y) {
		parent[Find(x)] = Find(y);
	}
};

//TODO(luisvasquez) review if it worths to use ranked unionFind. It needs more testing.







.


