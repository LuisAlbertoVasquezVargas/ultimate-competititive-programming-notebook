
void sieve() {
	P = vInt(N + 1, -1); // TODO(luisvasquez): to rename p, and handle N ass parameter.
	P[0] = P[1] = 0;
	for (int i = 2; i * i <= N; ++i) {
		if (P[i] == -1) {
			for (int j = i * i; j <= N; j += i) {
				P[j] = i;
			}
		}
	}
}

void primefact(int n, vInt &pr, vInt &ex) {
	while(true) {
		int p = P[n];
		if (n == 1) {
			return;
		}
		
		if (p == -1) {
			pr.push_back(n);
			ex.push_back(1);
			break;
		}
		int exp = 0;
		while (n % p == 0) {
			exp ++;
			n /= p;
		}
		pr.push_back(p);
		ex.push_back(exp);
	}
}

vInt getAllDivisors(vInt pr, vInt ex) {
	int k = SZ(pr);
	vInt divisors(1, 1);

	REP (i, k) {
		int m = SZ(divisors);
		REP (j , ex[i]) {
			REP (a, m) {
				divisors.push_back(pr[i] * divisors[a + j * m]);
			}
		}
	}
	return divisors;
}


