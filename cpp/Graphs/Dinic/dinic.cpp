#define INF (1 << 30)
#define N 200000

#define DEBUG(x) cout << #x << " " << (x) << endl;

typedef vector<int> vi;
typedef long long ll;

struct flowGraph {
  int n, m, s, t, E;
  vi to, cap, NEXT;    // maxe * 2
  vi last, now, dist;  // maxv
  flowGraph() {}
  flowGraph(int n, int m, int s, int t) : n(n), m(m), s(s), t(t) {
    to = cap = NEXT = vi(2 * m + 5);
    last = now = dist = vi(n + 5);
    E = 0;
    last = vi(n + 5, -1);
  }
  void add(int u, int v, int uv, int vu = 0) {
    to[E] = v;
    cap[E] = uv;
    NEXT[E] = last[u];
    last[u] = E++;
    
    to[E] = u;
    cap[E] = vu;
    NEXT[E] = last[v];
    last[v] = E++;
  }
  bool bfs() {
    REP (i, n)
      dist[i] = INF;
    queue<int> Q;
    dist[t] = 0;
    Q.push(t);
    while (!Q.empty()) {
      int u = Q.front();
      Q.pop();
      for (int e = last[u]; e != -1; e = NEXT[e]) {
        int v = to[e];
        if (cap[e ^ 1] && dist[v] >= INF) {
          dist[v] = dist[u] + 1;
          Q.push(v);
        }
      }
    }
    return dist[s] < INF;
  }
  int dfs(int u, int f) {
    if (u == t) return f;
    for (int &e = now[u]; e != -1; e = NEXT[e]) {
      int v = to[e];
      if (cap[e] && dist[u] == dist[v] + 1) {
        int ret = dfs(v, min(f, cap[e]));
        if (ret) {
          cap[e] -= ret;
          cap[e ^ 1] += ret;
          return ret;
        }
      }
    }
    return 0;
  }
  ll maxFlow() {
    ll flow = 0;
    while (bfs()) {
      REP (i, n)
        now[i] = last[i];
      while (1) {
        int f = dfs(s, INF);
        if (!f) break;
        flow += f;
      }
    }
    return flow;
  }
};