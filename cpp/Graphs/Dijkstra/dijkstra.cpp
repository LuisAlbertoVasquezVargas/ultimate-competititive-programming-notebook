// TODO(luisvasquez): to put struct Node inside DirectedGraph
// TODO(luisvasquez): to try to avoid the use of struct Node
// TODO(luisvasquez): to put more versions that are faster like when only need to get a target node

const Long INF = (1LL << 61);
const int N = 1e5;

struct Node {
    int id;
    Long dist;
    Node(int id, Long dist): id(id), dist(dist) {}
};

bool operator < (const Node &a, const Node &b) { 
    return a.dist > b.dist; 
}

struct DirectedGraph {
    int n;
    vvInt G;
    vvInt C;
    DirectedGraph(int n, vvInt &edges) : n(n) {
        G.resize(n);
        C.resize(n);
        for (auto edge:edges) {
            int u = edge[0];
            int v = edge[1];
            int w = edge[2];
            add(u, v, w);
        }
    }

    void add (int u, int v, int w) {
        G[u].PB(v);
        C[u].PB(w);
    }

    vLong dijkstra (int s) {
        vLong dist(n, INF);
        vInt vis(n);
        priority_queue<Node> q;
        dist[s] = 0;
        q.push(Node(s, 0));
        while (!q.empty()) {
            Node pu = q.top();
            q.pop();
            int u = pu.id;
            if (vis[u]) {
                continue;
            }
            vis[u] = true;
            REP (i, SZ(G[u])) {
                int v = G[u][i];
                int w = C[u][i];
                if (!vis[v] && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    q.push(Node(v, dist[v]));
                }
            }
        }
        return dist;
    }
};
