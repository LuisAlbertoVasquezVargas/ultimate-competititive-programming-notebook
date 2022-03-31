// Problem
// https://leetcode.com/problems/largest-color-value-in-a-directed-graph/

enum {WHITE, GRAY, BLACK};

vvInt G;
vInt vis;
bool hasCycle;
void init(int n, vvInt &edges) {
    G = vvInt(n);
    vis = vInt(n, WHITE);
    hasCycle = false;
    for (vInt &edge:edges) {
        int u = edge[0];
        int v = edge[1];
        G[u].PB(v);
    }
}
void dfs(int u) {
    vis[u] = GRAY;
    for (int v:G[u]) {
        if (vis[v] == BLACK) {
            continue;
        }
        if (vis[v] == WHITE) {
            dfs(v);
        } else {
            hasCycle = true;
        }
    }
    vis[u] = BLACK;
}

void solve(int n, vector<vector<int>>& edges) {
    int n = SZ(colors);
    init(n, edges);
    REP (u, n) {
        if (vis[u] == WHITE) {
            dfs(u);
        }
    }
}