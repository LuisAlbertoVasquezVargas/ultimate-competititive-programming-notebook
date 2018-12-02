struct UnionFind {
	vInt pa; // TODO(luisvasquez) : search another name more meaningful.
  void initUF(int n) {
  	pa = vInt(n);
  	REP (i, n) {
  		pa[i] = i;
  	}
  }
  
 // TODO(luisvasquez): investigate if can use find, union functions names. 
  int Find( int x ){ 
		return pa[ x ] = (pa[ x ] == x ? x : Find(pa[ x ]) );
  } 
	void Union(int x, int y) {
		pa[Find(x)] = Find(y);
	} // 
};

//TODO(luisvasquez) review if it worths to use ranked unionFind.


