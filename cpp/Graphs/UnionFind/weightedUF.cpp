// Problem:
// https://leetcode.com/problems/maximum-subarray-min-product/
// implementation adapted from 
// https://cp-algorithms.com/data_structures/disjoint_set_union.html

struct UnionFind {
  vInt parent;
  vInt size;
  vLong weight;
  UnionFind(int n, vInt &w) {
    parent.resize(n);
    size.resize(n, 1);
    weight.resize(n);
    REP (i, n) {
        weight[i] = w[i];
    }
    iota(ALL(parent), 0);
  }

  int find(int x) { 
    return parent[x] = (parent[x] == x ? x : find(parent[x])); 
  }

  void unite(int a, int b) { 
    a = find(a);
    b = find(b);
    if (a != b) {
        if (size[a] < size[b])
            swap(a, b);
        parent[b] = a;
        size[a] += size[b];
        weight[a] += weight[b];
    }
  }
};