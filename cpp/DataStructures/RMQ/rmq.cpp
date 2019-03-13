struct RMQ {
    vvInt rmq;
    RMQ() {}
    RMQ(vInt &A) {
        int n = SZ(A);
        int LOGN = 0;
        // TODO(luisvasquez) : to change the log for something like a bitwise operation or math formula.
        while ((1 << LOGN) < n) {
            LOGN ++;
        }
        rmq = vvInt(LOGN + 1, vInt(n));
        // TODO(luisvasquez) : to change it just to rmq[0] = A
        REP (i, n) { 
            rmq[0][i] = A[i]; 
        }
        
        for (int k = 1; k <= LOGN; ++k) {
            for (int i = 0; i + (1 << k) <= n; ++i) {
                rmq[k][i] = min(rmq[k - 1][i], rmq[k - 1][i + (1 << (k - 1))]);
            }
        }
    }

    int query(int a, int b) {
        int r = 31 - __builtin_clz(b - a + 1);
        return min(rmq[r][a], rmq[r][b - (1 << r) + 1]);
    }
};
