struct flowGraph {
  int n, m, s, t, E;
  const int INF = INT_MAX / 2;
  vInt to, cap, NEXT;    // maxe * 2
  vInt last, now, dist;  // maxv
  flowGraph() {}
  flowGraph(int n, int m, int s, int t) : n(n), m(m), s(s), t(t) {
    // TODO(luisvasquez) : to investigate about NEXT variable.
    to = cap = NEXT = vInt(2 * m + 5);
    last = now = dist = vInt(n + 5);
    E = 0;
    last = vInt(n + 5, -1);
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
    REP (i, n) {
      dist[i] = INF;
    }
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
  Long maxFlow() {
    Long flow = 0;
    while (bfs()) {
      REP (i, n) {
        now[i] = last[i];
      }
      while (true) {
        int f = dfs(s, INF);
        if (!f) {
          break;
        }
        flow += f;
      }
    }
    return flow;
  }
};