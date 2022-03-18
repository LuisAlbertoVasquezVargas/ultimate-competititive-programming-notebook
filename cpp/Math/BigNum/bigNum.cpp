#include<bits/stdc++.h>
using namespace std;

typedef long long Long;
typedef vector<int> vInt;

#define REP(i,n) for(int i = 0; i < n; i++)
#define ALL(v) v.begin(), v.end()
#define SZ(v) (int)v.size()
#define PB push_back

// TODO(luisvasquez): to change printf to cout Or parametrize with a class/struct returning strings

const int base = 1000 * 1000 * 1000;
void impr(vInt &a) {
	printf("%d", a.empty() ? 0 : a.back());
	for (int i = SZ(a) - 2; i >= 0; --i)
		printf ("%09d", a[i]);
    cout << endl;
}

int getLen(vInt &a) {
	if (a.empty()) return 1;
	int r = a.back();
	int ans = 0;
	while (r) {
		r /= 10;
		ans ++;
	}
	ans += 9 * (SZ(a) - 1);
	return ans;
}

vInt get(string s) {
	vInt a;
	for (int i = (int)s.length(); i > 0; i -= 9)
		if (i < 9)
			a.push_back (atoi (s.substr(0, i).c_str()));
		else
			a.push_back (atoi (s.substr(i-9, 9).c_str()));
	while (a.size() > 1 && a.back() == 0) a.pop_back();
	return a;
}

vInt sum(vInt a, vInt b) {
	int carry = 0;
	for (size_t i = 0; i < max(a.size(), b.size()) || carry; ++i) {
		if (i == a.size())
			a.push_back(0);
		a[i] += carry + (i < b.size() ? b[i] : 0);
		carry = a[i] >= base;
		if (carry)  a[i] -= base;
	}
	return a;
}

vInt rest(vInt a, vInt b) {
	int carry = 0;
	for (size_t i = 0; i < b.size() || carry; ++i) {
		a[i] -= carry + (i < b.size() ? b[i] : 0);
		carry = a[i] < 0;
		if (carry)  a[i] += base;
	}
	while (a.size() > 1 && a.back() == 0)
		a.pop_back();
	return a;
}

vInt mult1(vInt a, int b) {
	int carry = 0;
	for (size_t i = 0; i < a.size() || carry; ++i) {
		if (i == a.size())
			a.push_back(0);
		long long cur = carry + a[i] * 1ll * b;
		a[i] = int (cur % base);
		carry = int (cur / base);
	}
	while (a.size() > 1 && a.back() == 0)
		a.pop_back();
	return a;
}

vInt mult2(vInt a, vInt b) {
	vInt c (a.size() + b.size());
	for (size_t i = 0; i < a.size(); ++i)
		for (int j = 0, carry = 0; j < (int)b.size() || carry; ++j) {
			long long cur = c[i + j] + a[i] * 1ll * (j < (int)b.size() ? b[j] : 0) + carry;
			c[i+j] = int (cur % base);
			carry = int (cur / base);
		}
	while (c.size() > 1 && c.back() == 0)
		c.pop_back();
	return c;
}

vInt divide(vInt a, int b, int &car) {
	int carry = 0;
	for (int i = (int)a.size() - 1; i >= 0; --i) {
		long long cur = a[i] + carry * 1ll * base;
		a[i] = int (cur / b);
		carry = int (cur % b);
	}
	while (a.size() > 1 && a.back() == 0)
		a.pop_back();
	car = carry;
	return a;
}