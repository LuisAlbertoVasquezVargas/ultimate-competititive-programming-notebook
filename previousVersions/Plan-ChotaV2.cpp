/*
	### INDICE ###
	* TEMPLATE 
		- Librerias
		- Procesamiento de input
		- Shorcuts ( #define )
	* UTILITIES
		- getchar_unlocked
		- freopen
		- toi , tos ( toInt and toString )
		- sync
	* MATH - NUMBER THEORY	
		- fast exponentiation( iterative ) + modular inverse	O( log n )
		- primality test
			- Simple - Deterministic  							O( sqrt n ) 
			- Miller_Rabin - Probabilistic 						O( it log(n)^3 )
		- sieve
			- primes , +fact +generating all divisors
			- phi
		- combinatorics and factorial
			 Multiplicative formula								O( n/2 )
			 Recursive formula									O( n^2 )
			 Factorial formula									O( n )
			 Lucas Teorem 										O( p + Qlog n )
			 Generalized lucas theorem
		- gcd & lcm ( recursive ) 								O( log n )
		- extended gcd											O( log n )
		- josephus												//REVISAR !!
		- phi 													O( sqrt n )
		- factorizacion prima
			- Deterministc 										O( sqrt n )
			- Probabilistic ( Pollard Rho ) 
		- Teorema chino del resto
		- FFT fast fourier transformation 
		- Karatsuba 
	* DP
		- N - Tiling
		- Longest Common Subsequence ( LCS )					O( nm )
		- Coin Change											O( nV )
		- Edit Distance											O( nm )
		- Max 1D Range Sum										O( n )
		- Table additive 2D , 3D ( Cuentita )					O( n^2 , n^3 )
		- Subset Sum											O( S*n )
		- DP over digits( dp peligro o bool peligro )			O( digits )//
		- Longest increasing subsequence ( LIS )				O( nlogn )//
		- Value Spected
		- DP in dag
	* GRAPHS
		- Union Find 
			- union find
			- union find by rank( with path compression )
			- Edge( Kruskal Algorithm )
		- Djkstra
		- Euler Tour
		- Cycle Detection
		- LCA ( By peluche )									O( nlogn + Qlogn )
		- Floyd Warshall										O( n^3 )
		- Topological_Sort
		- SCC of Tarjan											O( E + V )
		- BCC ( Chen )
		- Bicoloring
		- Heavy Light Decomposition HLD
		- Centroid Decomposition
		- Bellman Ford
		- min_cost_arborescence
		- eulerian path
		- minimum distance spanning tree
	* FLOWS
		- Bipartite Matching
			- BIPARTITE MATCHING ( sonnycson )					O( E V )
				- Short Code									O( E V )
				- Insert and erase Edges						O( E V )
			- BPM Hopcroft Karp ( Fastest by Shinta )			O( E sqrt V )
			- NON - BIPARTITE MATCHING (CHEN)
		- MAX FLOW
			- FORD - FULKERSON									O( E max| f | )
			- DINIC												O( E sqrt V )
			- MAX_FLOW_MIN_COST
			- HUNGARIAN ALGORITHM 								O( n^3 )
			- ALL PAIRS MAXMIMUM FLOW
	* STRINGS
		- Knuth Morris Pratt algorithm ( KMP )					O( n + m )
		- Trie													O( ALF*C )
		- Aho Corasick											O( C )
		- Hashing												----
		- Manacher algorithm
		- Suffix Array											O( n logn^2 )
		- StringUtilities( Chen )
		- Z Algorithm
	* MATRICES
		- fast matrix exponentiation							O( n^3log( k ) )
		- gaussian elimination 
			-( Invert and find determinant of Matrix )			O( n^3 )
		- Gaussian Eliminaton MOD
		- matrix rank mod 2 
	* DATA STRUCTURES
		- Segment Tree  										O( n + Qlogn )
			Lazy Propagation - 2D( E-maxx )	
			Persistent segment tree , ST with pointers	
		- Sparse table (Range Minimun Query ( RMQ )	)			O( nlogn + Q )
		- Sliding RMQ
		- Sqrt Decomposition 									O( n*sqrt(n) + q*sqrt(n) )
		- MergeSort												O( nlogn )
		- CountInversions										O( nlogn^2 ) y O( nlogn ) 
		- Treap( Roy - Emaxx )									O( logn )
		- Bit 1D y 2D + Dquery 									O( log( n ) )
	* GEOMETRIA COMPUTACIONAL
		- Version Resumida
		- Long double Version
			- arg get_angle norm unit cross dot area dist isParallel lineIntersection 
			- Bis distPointLine CircumscribedCircle InscribedCircle TangentLineThroughPoint  
			- CircleThroughAPointAndTangentToALineWithRadius CircleTangentToTwoLinesWithRadius 
			- CircleTangentToTwoDisjointCirclesWithRadius area pointInPoly
		- Representacion Implicita y Vectorial de Lineas
		- Version de Roy
		 	- dist cross dot area areaHeron circumradius areaHeron circumradius between onSegment 
		 	- intersects sameLine isParallel lineIntersection circumcenter isConvex area pointInPoly 
		 	- TEOREMA DE PICK  ConvexHull isInConvex(log n) isInConvex SMALLEST ENCLOSING CIRCLE O(n) 
		 	- CLOSEST PAIR OF POINTS INTERSECCION DE CIRCULOS circleCircleIntersection LINEA AB vs CIRCULO (O, r) 
		 	- lineCircleIntersection CircumscribedCircle InscribedCircle TangentLineThroughPoint 
		 	- TangentLineThroughPoint CircleThroughAPointAndTangentToALineWithRadius 
		 	- CircleTangentToTwoLinesWithRadius CircleTangentToTwoDisjointCirclesWithRadius 
		 	- Rotating Callipers ( AntipodalPairs )
			- Spherical Geometry
			- Rotation Matrix
		- sweep line
			- Union de Intervalos
			- Largest empty Rectangle
			- agregar closest pair
	*JAVA TEMPLATE
	* OTHERS
		- Perfect Hashing
		- Permutation
		- BIG NUMS( by roy )
		- STABLE MARRIAGE
		- Sudoku
		- Count Digits
		- Roman Numbers ( arabic to roman )
		- union de intervalos
		- TERNARY SEARCH
		- ConvexHull Trick
		- BAIRSTOW METHOD AND POLYNOMIALS
		- Big num ( by chen )
		- (agregar) compresion de coordenadas
		- (agregar) calculadora - EQUIVALENCE ACM 5872
		- (agregar) polinomios de varias variables - Equivalence
		- TWO POINTERS
		- GRUNDY NUMBERS
*/
###################################################################
########################## TEMPLATE #################################
###################################################################
////////////////////////////////////Libs////////////////////////////////////////////////////////////////////////
#include <vector>
#include <list>
#include <map>
#include <set>
#include <queue>
#include <deque>
#include <stack>
#include <bitset>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <climits>
#include <cctype>
////////////////////////////////////////////////////////////////////////
#include <bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; i++ )
#define clr( t , val ) memset( t , val , sizeof(t) )

#define all(v)  v.begin() , v.end()
#define pb push_back
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define test() cerr<<"hola que hace ?"<<endl;
#define DEBUG(x) cerr<<#x<<"="<<x<<endl;
#define DEBUG2(x,y) cerr<<#x<<"="<<x<<" "<<#y<<"="<<y<<endl;
#define DEBUG3(x,y,z) cerr<<#x<<"="<<x<<" "<<#y<<"="<<y<<" "<<#z<<"="<<z<<endl;

typedef long long ll;
typedef pair< int , int > pii;
typedef vector< int > vi;

int main(){

}
/////
#define y0 sdkfaslhagaklsldk
#define y1 aasdfasdfasdf
#define yn askfhwqriuperikldjk
#define j1 assdgsdgasghsf
#define tm sdfjahlfasfh
#define lr asgasgash
///
###################################################################
################# UTILITIES #######################################
###################################################################
// 1 millon de enteros = 4 mb
// 1mb = 1024 K
################# GETCHAR_UNLOCKED #######################################
#define getcx getchar_unlocked
 
using namespace std;
 
inline void read(int &number) {
        number = 0;
        int ch = getcx();
        while (ch < '0' || ch > '9')
                ch = getcx();
        while(ch >= '0' && ch <= '9')
                number = (number << 3) + (number << 1) + ch - '0', ch = getcx();
}

#-----------------------------------------------------------------#
################# FREOPEN #######################################
#include<cstdio>
freopen("input.txt", "r", stdin);
freopen("output.txt", "w", stdout);
#-----------------------------------------------------------------#
#################### TOI y TOS #######################################
// Obs a veces es mejor hacer un while en lugar de transformar a cadena
ll toi(string s){istringstream is(s);ll x;is>>x;return x;}
string tos(ll t){stringstream st; st<<t;return st.str();}
//( transform( all(t) , t.begin() , ::tolower );
#-----------------------------------------------------------------#
########################## SYNC #################################
// Para acelerar la lectura de strings cuando se usa cin
/*
	Experimentalmente con cadenas
	LOCAL COMPILER
	scanf	cin + sync	cin 
 n = 10^7			
	0.100		0.132	1.825
 n = 10^6			
	0.017		0.024	0.202
 n = 10^5			
 	0.002		0.003	0.031
// Probar con diversos jueces Timus , Codeforces , SPOJ etc .
*/
// Probar con io::sync_with_stdio(false);
#define sync ios_base::sync_with_stdio(false);
ios_base::sync_with_stdio(false);
#-----------------------------------------------------------------#

###################################################################
################# MATH - NUMBER THEORY #######################################
###################################################################

########################## FAST EXPONENTIATION ##################################
// Simple version O( log(n) ) - iterative
// UVA 374 - Big Mod
// agregar version recursiva
ll pow( ll a , ll b , ll c ){
    ll ans = 1;	
    while( b ){
        if( b&1 ) ans = ( ans * a )%c;
        a = ( a*a )%c;
        b >>= 1;
    }	
    return ans;
}
// Inverse modular de t es pow( t ,MOD - 2 ,  MOD )
// http://chococontest.wordpress.com/category/inverso-modular/

ll mod_inv( ll numa , ll mod ){	return pow( numa , mod - 2 , mod );	}

// A^B mod C Problem  (1<=A,B,C<2^63). 1752_FZU
// a += b   %c
void fix( ull &a , ull b , ull c ){
	a += b;
	if( a >= c ) a -= c;
}
ull prod( ull a , ull b , ull c ){
	a %= c;
	ull ans = 0;
	while( b ){
		if( b & 1 ) fix( ans , a , c );
		a <<= 1;
		if( a >= c ) a -= c;
		b >>= 1;
	} 
	return ans;
}
ull pow( ull a , ull b , ull c ){
	ull ans = 1;
	while( b ){
		if( b & 1 ) ans = prod( ans , a , c );
		a = prod( a  , a , c  );
		b >>= 1;
	}
	return ans;
}

#-----------------------------------------------------------------#
######################### PRIMALITY TEST ##################################

//	Simple - Deterministic ( O( sqrt(num) ) )
// UVA 10924 - Prime Words ... cuidado considera primo el 1 (buscar otro probl)
bool isPrime( int n ){
	if( n == 1 ) return 0;
	for( int i = 2 ; i * i <= n ; ++ i )
		if( n%i == 0 )return 0;
	return 1;
}
bool isPrime( int n )
{
	if( n <= 2 ) return n == 2;
	if( !( n&1 ) ) return 0;
	for( int i = 3 ; i*i <= n ; i += 2 )
		if( n%i == 0 ) return 0;
	return 1;
}

####################### Miller-Rabin ####################################

// Prime Gap http://en.wikipedia.org/wiki/Prime_gap  O( ln(n)^2 )
// By Chen : Test de primalidad para numeros grandes O(it*log(n)^3)

////////// Version 1 para n <= 10^9

int pow( int a , int b , int c ){
    int ans = 1;	
    while( b ){
        if( b&1 ) ans = ( 1LL*ans * a )%c;
        a = ( 1LL*a*a )%c;
        b >>= 1;
    }	
    return ans;
}
bool miller(ll p, int it = 10){
	if(p < 2) return 0;
	if(p != 2 && (p & 1) == 0) return 0;
	int s = p - 1;
	while( (s & 1) == 0) s >>= 1;
	while( it-- ){
		int a = rand() % (p - 1) + 1, temp = s;
		int mod = pow(a, temp, p);
		while(temp != p - 1 && mod != 1 && mod != p - 1){
			mod = ( 1LL* mod*mod )%p;
			temp <<= 1;
		}
		if(mod != p - 1 && (temp & 1) == 0) return 0;
	}
	return 1;
}
///// Version 2 con multiplicaciones eficientes( falta analizar ^^ )

ll mulmod(ll a, ll b, ll c){
	ll x = 0, y = a % c;
	while(b > 0){
		if(b & 1){
			x += y;
			if(x >= c) x -= c;
		}
		y = y + y;
		if(y >= c) y -= c;
		b >>= 1;
	}
	return x % c;
}
ll modulo(ll a, ll b, ll c){
	ll x = 1, y = a;
	while(b){
		if(b & 1) x = mulmod(x, y, c);
		y = mulmod(y, y, c);
		b >>= 1;
	}
	return x % c;
}
bool miller(ll p, int it = 20){
	if(p < 2) return 0;
	if(p != 2 && (p & 1) == 0) return 0;
	ll s = p - 1;
	while( (s & 1) == 0) s >>= 1;
	while( it-- ){
		ll a = rand() % (p - 1) + 1, temp = s;
		ll mod = modulo(a, temp, p);
		while(temp != p - 1 && mod != 1 && mod != p - 1){
			mod = mulmod(mod, mod, p);
			temp <<= 1;
		}
		if(mod != p - 1 && temp % 2 == 0) return 0;
	}
	return 1;
}
################## SIEVE O(nlglgn) #########################################
//replace this implementation with bitset and factoring primes i dont know :P
const int MAX_N=100000;
bool prime[MAX_N + 1];

void sieve(){
	memset(prime, true, sizeof(prime));
	prime[0]=false;
	prime[1]=false;
	for(int i=2;i*i<=MAX_N;i++)
		if(prime[i])
			for(int j=i*i;j<=MAX_N;j+=i)
				prime[j]=false;
}

//http://apps.topcoder.com/forums/?module=Thread&threadID=594206&start=0&mc=25
//generate all divisors
p is an array of primes
e is the exponent on each of those primes
k is the number of elements in p and e

Basically the idea is to calculate all the previous divisors * p, then use those values to calculate all previous divisors * p^2, etc... I could just keep track of p[i]^j but this way ends up being a bit shorter/cleaner.
//////////////////////////////

int P[ N + 5 ];
void sieve(){
	clr( P , -1 );
	P[ 0 ] = P[ 1 ] = 0;
	for( int i = 2 ; i * i <= N ; ++i )
		if( P[ i ] == -1 )
			for( int j = i * i ; j <= N ; j += i ) P[ j ] = i;
}
void primefact( int n , vi &pr , vi &ex ){
	while( 1 ){
		int p = P[ n ];
		if( n == 1 ){
			return;
		}
		if( p == -1 ){
			pr.pb( n ) , ex.pb( 1 );
			break;
		}
		int exp = 0 ;
		while( n % p == 0 ) {
			exp ++;
			n /= p;
		}
		pr.pb( p ) , ex.pb( exp );
	}
}

vi getAllDivisors( vi pr , vi ex ){
	int k = pr.size();
	vi divisors( 1 , 1 );

	REP( i , k ){
		int m = divisors.size();
		REP( j , ex[ i ] )
			REP( a , m ) divisors.pb( pr[ i ] * divisors[ a + j * m ] );
	}
	return divisors;
}

////////////////////////////////////////////////////////////////////////////
#define MAXN 300000000

bitset <MAXN+1> notprime;

for(int i=3; i*i<=MAXN; i+=2)
	if(!notprime[i])
		for(int j=i*i; j<=MAXN; j+=(i<<1))
			notprime[j] = true;
				
#-----------------------------------------------------------------#

################## SIEVE O(n) #########################################

//Falta agregar factorizacion con la criba
// Comentarios generales :
// p[i] para 0 < i indiva el valor del primo i-esimo 
//	Ejm : p[1] = 2 , p[2] = 3 ....
// A[i] indica que el menor factor primo de i es el primo A[i] - esimo 
//	Ejm: si 15 = 3*5 , entonces A[12] = 2 porque el menor factor primo de 12 es 3 y 3 es el 2do primo
#define MAXN 100000
int A[MAXN + 1], p[MAXN + 1], pc = 0;  
void sieve()  
{ 
    for(int i=2; i<=MAXN; i++){ 
        if(!A[i]) p[A[i] = ++pc] = i; 
        for(int j=1; j<=A[i] && i*p[j]<=MAXN; j++) 
            A[i*p[j]] = j; 
    } 
} 
#-----------------------------------------------------------------# 

######################## funcion PHI de euler ###################################

#define MAXN 10000
int phi[MAXN + 1]
for(i = 1; i <= MAXN; ++i) phi[i] = i;
for(i = 1; i <= MAXN; ++i) for (j = i * 2; j <= MAXN; j += i) phi[j] -= phi[i];

#define MAXN 3000000
int phi[MAXN + 1], prime[MAXN/10], sz;
bitset <MAXN + 1> mark;

for (int i = 2; i <= MAXN; i++ ){
	if(!mark[i]){
		phi[i] = i-1;
		prime[sz++]= i;
	}
	for (int j=0; j<sz && prime[j]*i <= MAXN; j++ ){
		mark[prime[j]*i]=1;
		if(i%prime[j]==0){
			phi[i*prime[j]] = phi[i]*prime[j];
			break;
		}
		else phi[i*prime[j]] = phi[i]*(prime[j]-1 );
	}
}

#-----------------------------------------------------------------#


################### COMBINATORICS AND FACTORIALS ########################################
//http://en.wikipedia.org/wiki/Binomial_coefficient
/* 	
	C( n , k ) = C( n - 1 , k - 1 ) + C( n - 1 , k ) , C( n , n ) = C( n , 0 ) = 1
 	C( n , k ) = n!/(k!*(n-k)!)
 	C( n , k ) = C( n , n - k )
 	Propiedad 
		Sumatoria ( n , i )  0 <= i <= n = 2^n
	 F( n ) = n * F( n - 1 ) = n! , F[0] = 1
	 FP( n ) = 1/F(n) = ( n + 1 )/[ F( n ) * ( n + 1 ) ] = ( n + 1 )/ F( n + 1 ) = ( n + 1 )*FP( n + 1 ) 
*/
//Multiplicative formula
// Complejidad O( n/2 )
// UVA 369 - Combinations
// Muy buena version para hallar valores exactos del comb
ll comb( int n , int k )
{
    if( k > n - k ) k = n - k;
    ll C = 1;
    REP( i , k ) C = C * ( n - i )/( 1 + i );
    return C;
}
//Recursive formula
// Complejidad O( n^2 )
// Muy buena version para hallar muchos querys
#define N 100

ll C[ N + 5 ][ N + 5 ];
void init( int n ){
    clr( C, 0 );
	REP( i , n + 1 ) C[ i ][ i ] = C[ i ][ 0 ] = 1;
    for( int i = 2 ; i <= n ; ++i )
         for( int j = 1 ; j <= i ; ++j )
            C[ i ][ j ] = ( C[ i - 1 ][ j ] + C[ i - 1 ][ j - 1 ] );
}
//Factorial formula
// Version muy util para hallar C( n , m ) % MOD
// Version muy eficiente con inverso modular , n  , m <= 10^6
// 3412 TJU
#define N 10000
#define MOD 1000000007LL

ll F[ N + 5 ] , FP[ N + 5 ];

ll pow( ll a , ll b , ll c ){
	ll ans = 1;
	while( b ){
		if( b & 1 ) ans = ( ans * a )%c;
		a = ( a * a )%c;
		b >>= 1;
	}
	return ans;
}
ll mod_inv( ll a , ll p ){
	return pow( a , p - 2 , p );
}
void init(){
	F[ 0 ] = 1;
	for( int i = 1 ; i <= N ; ++i ) 
		F[ i ] = ( (ll) i * F[ i - 1 ] )%MOD;
	FP[ N ] = mod_inv( F[ N ] , MOD ) ;
	for( int i = N - 1 ; i >= 0 ; --i ) 
		FP[ i ] = ( (ll)FP[ i + 1 ]*( i + 1 ) )%MOD;
}
// Lucas Teorem 
//http://en.wikipedia.org/wiki/Lucas'_theorem
// Cuando n y m son grandes y se pide comb(n,m)%MOD, donde MOD es un numero primo, se puede usar el Teorema de Lucas.
// explicar lucas
ll comb( ll n, ll k ){
	ll ans = 1;
    while( n > 0 ){
        ll ni = n%MOD,ki = k%MOD;
        n /= MOD; k /= MOD;
        if( ni - ki < 0 )return 0;
        ll temp = (FP[ki]*FP[ni-ki])%MOD;
        temp = (temp*F[ni])%MOD;
        ans = (ans*temp)%MOD;
    }
    return ans;
}
#-----------------------------------------------------------------#
// Generalized lucas theorem
//http://codeforces.com/blog/entry/10271
//http://codeforces.com/gym/100637/problem/D
struct EuclidReturn{
    ll u , v , d;
    EuclidReturn( ll u , ll v, ll d ) : u( u ) , v( v ) , d( d ) {}
};
    
EuclidReturn Extended_Euclid( ll a , ll b){
    if( b == 0 ) return EuclidReturn( 1 , 0 , a );
    EuclidReturn aux = Extended_Euclid( b , a%b );
    ll v = aux.u - (a/b)*aux.v;
    return EuclidReturn( aux.v , v , aux.d );
}

// ax = 1(mod n)
ll modular_inverse( ll a , ll n ){
    EuclidReturn aux = Extended_Euclid( a , n );
    return ((aux.u/aux.d)%n+n)%n;
}

ll chinese_remainder( vll &rem, vll &mod ){
	ll ans = rem[ 0 ] , m = mod[ 0 ];
    for( int i = 1 ; i < SZ(rem) ; ++i ){
        int a = modular_inverse( m , mod[ i ] );
        int b = modular_inverse( mod[ i ] , m );
        ans = ( ans * b * mod[ i ] + rem[ i ] * a * m)%( m*mod[ i ] );
        m *= mod[i];
    }
    return ans;
}


void primefact( int n , vll &p , vll &e , vll &pe ){
	for( int i = 2 ; i * i <= n ; ++i ){
		if( n % i == 0 ){
			int exp = 0 , pot = 1;
			while( n % i == 0 ){
				n /= i;
				exp ++;
				pot *= i;
			}
			p.pb( i ) , e.pb( exp ) , pe.pb( pot );
		}
	}
	if( n > 1 ) p.pb( n ) , e.pb( 1 ) , pe.pb( n );
}

ll pow( ll a , ll b , ll c ){
	ll ans = 1;
	while( b ){
		if( b & 1 ) ans = (ans * a)%c;
		a = (a * a)%c;
		b >>= 1;
	}
	return ans;
}
ll factmod( ll n , ll p , ll pe ){
	if( n == 0 ) return 1;
	ll cpa = 1;
    ll ost = 1;
    for( ll i = 1; i <= pe; i++ ){
        if( i % p != 0 ) cpa = (cpa * i) % pe;
        if( i == (n % pe) ) ost = cpa;
    }
    cpa = pow(cpa, n / pe, pe);
    cpa = (cpa * ost) % pe;
    ost = factmod(n / p, p, pe);
    cpa = (cpa * ost) % pe;
    return cpa;
}
ll factst( ll a , ll b ){
	ll ans = 0;
	while( a ){
		ans += a / b;
		a /= b;
	}
	return ans;
}

ll solve( ll n , ll k , ll p , ll e , ll pe ){
	ll np = factmod( n , p , pe );
	ll kp = factmod( k , p , pe );
	ll nkp = factmod( n - k , p , pe );
	ll cnt = factst( n , p ) - factst( k , p ) - factst( n - k , p );
	if( cnt >= e ) return 0;
	ll r = ((np * modular_inverse( kp , pe ))% pe);
	r = (r * modular_inverse( nkp , pe ))%pe;
	REP( i , cnt ) r = (r * p) % pe;
	return r;
}

int main(){
	ll n , k , mod;
	while( cin >> n >> k >> mod ){
		vll p , e , pe;// pe = p ^ e
		primefact( mod , p , e , pe );
		vll rem;
		REP( i , SZ( p ) ) rem.pb( solve( n , k , p[ i ] , e[ i ] , pe[ i ] ) );
		cout << chinese_remainder( rem ,  pe ) << '\n';
	}
}

////
//http://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
//http://en.wikipedia.org/wiki/Multinomial_theorem
########################## GCD & LCM #################################

/// FALTA Implementar version iterativa del GCD
/// Complejidad O(log(n)) n = max(a,b)
// Elemento neutro del GCD == 0  si y solo si gcd( 0 , x ) = gcd( x , 0 ) = x 

ll GCD( ll a , ll b ){
	if(b == 0) return a;
	return GCD( b , a%b );
}

ll lcm( ll a, ll b){
	return a*b/GCD( a,  b );
}
/// LCM(a1,a2,...ak) = LCM(a1,LCM(a2,a3,...,ak)) = LCM(a1,LCM(a2,LCM(a3,a4,..ak)))
/// ANalogamente para el GCD

ll LCM( ll a , ll b , ll c ){
	return lcm(a,lcm(b,c));	
}
########################## EXTENDED GCD #################################

//Given A and B, use the extended Euclidean algorithm to determine not only gcd(A,B) but also the solutions (X, Y) to AX + BY = gcd(A,B).
//The extended Euclidean algorithm already guarantees that |X| + |Y| is minimal, and that X <= Y
// Number Theorical algorithms ( Dropbox )
// http://www.questtosolve.com/browse.php?vol=101
// UVA 10104 - Euclid Problem
// Bounds: |x|<=b+1, |y|<=a+1.
// Based on SU notebook

void gcdext( int &g , int &x , int &y , int a , int b ){// ASUMING a >= b
	if( b == 0 )
		g = a , x = 1 , y = 0;
	else gcdext( g , y , x , b , a%b ) , y = y - ( a/b )*x;
}
// From number theoretic algorithms .ppt

// ax = b (mod n)
// gcdext( g , x , y, a , n );
// if( b%g == 0 ) return x*( b/g ) ;
// return -1;

// ax = 1(mod n)
// gcdext( g , x , y, a , n );
// if( b%g == 0 ) return x/g ;

#-----------------------------------------------------------------#
#-----------------------------------------------------------------#
################### JOSEPHUS #########################################

// 1 - based O( n )
int joseph (int n, int k) {
	int res = 0;
	for (int i=1; i<=n; ++i)
		res = (res + k) % i;
	return res + 1;
}

//Falta analizar :(
//agregar version nlogn
int survivor(int num, int k ){
	int s,i;     
	for ( int s = 0 , i = 1 ; i <= num ; ++i ) s = ( s + k )%i;
	return ( s + 1 );
}
// Semantica
//Retorna el indice  (0-based) del sobreviviente # i  1 <= i <= n del Josephus Problem con tama\F1o inicial n y 
// saltos cada k  
// http://user.math.uzh.ch/halbeisen/publications/pdf/jos.pdf
int J( int n , int k , int i )
{
	/*
	Recursive
	if( i == 1 )return ( k - 1 )%n;
	return ( k + J( n - 1 , k , i - 1 ) )%n;
	*/
	int s = ( k - 1 )%( n - i + 1 ); 
	for( int I = 2 ; I <= i ; ++I )
		s = ( k + s )%( n + I - i );
	return s;
}
// J( n , 2 , n ) = 2( n - 2^( ceil(log2(n)) ) )
#-----------------------------------------------------------------#
######################## funcion PHI de euler ###################################

// F( n ) = n * Prod ( 1 - 1/p ) tal que p|n 
// Euler's totient or phi function, f(n) is an arithmetic function that counts the number of positive integers less than or equal to n that 
// are relatively prime to n. That is, if n is a positive integer, then f(n) is the number of integers k in the range 1 = k = n for which gcd(n, k) = 1.
//A000010		 Euler totient function phi(n): count numbers <= n and prime to n. 
// 1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16, 6, 18, 8, 12, 10, 22, 8, 20, 12, 18, 12, 28, 8, 30, 16, 20, 16, 24, 12, 36, 18, 24, 16, 40, 12, 42, 20, 24, 22, 46, 16, 42, 20, 32, 24, 52, 18, 40, 24, 36, 28, 58, 16, 60, 30, 36, 32, 48, 20, 66, 32, 44
// Simple version O( sqrt(n) )
int phi( int n ){
	int ans = n;
	for( int i = 2 ; i*i <= n ; ++i )
		if( n % i == 0 ){
			while( n%i == 0 ) n/= i;
			ans -= ans/i; 
		}
	if( n > 1 ) ans -= ans/n;
	return ans;
}
#-----------------------------------------------------------------#
############################## FACTORIZACION PRIMA #############################
// Deterministc
// 583  UVA
typedef vector< unsigned int > vi;

vi f( unsigned int n ){
	vi ans;
	while( n%2 == 0 ) ans.pb( 2 ) , n /= 2;
	for( int i = 3 ; i * i <= n ; i += 2 )
		if( n % i == 0 ){
			while( n%i == 0 ) ans.pb( i ) , n /= i;
		}
	if( n > 1 ) ans.pb( n );
	return ans;
}
##################################  Pollard Rho ####################################################
// Probabilistic
/// el algoritmo asume que num tiene al menos dos factores
// falta analizar la complejidad

// para n <= 10^9 o simplemente n^2 este entre en long long
typedef vector< int > vi;
typedef vector< vi > vvi;

typedef long long ll;
typedef unsigned long long ull;
typedef vector< ull > vull;

struct Pollard_Rho
{
	int q;
	vi v;	
	Pollard_Rho(){}
	Pollard_Rho( int x ) {
		q = x;
	}
	int mul( int a , int b , int c){
	    return ( (ll)a * (ll)b )%c;
	}
	int modd( int a , int b, int c){
	    int x = 1 , y = a; 
	    while( b ){
	        if( b&1 )
	            x = mul( x , y , c );
	        y = mul( y , y , c ); 
	        b >>=1 ;
	    }
	    return x;
	}
	bool Miller( int p,int iteration){
	    if( p < 2 )
	        return false;
	    if( p != 2 && (p&1) == 0 )
	        return false;
	    int s = p - 1;
	    while( (s&1) ==0)
	        s >>= 1;
	        
	    for(int i=0;i<iteration;i++){
	        int a=rand()%(p-1) + 1 , temp = s;
	        int mod = modd( a , temp , p );
	        while( temp != p - 1 && mod != 1 && mod != p - 1 ){
	            mod = mul( mod , mod , p );
	            temp <<= 1;
	        }
	        if( mod != p - 1 && (temp&1) == 0 )
	            return false;
	    }
	    return true;
	}
	int rho(int n){
	    if( (n & 1) == 0 ) return 2;
	    int x = 2 , y = 2 , d = 1;
	    int c = rand() % n + 1;
	    while( d == 1 ){
	        x = (mul( x , x , n ) + c)%n;
	        y = (mul( y , y , n ) + c)%n;
	        y = (mul( y , y , n ) + c)%n;
	        if( x - y >= 0 ) d = __gcd( x - y , n );
	        else d = __gcd( y - x , n );
	    }
	    return d;
	}
	void factor(int n){
	    if (n == 1) return;
	    if( Miller( n , 10 ) ){
	        if(q != n) v.push_back(n);
	        return;
	    }
	    int divisor = rho(n);
	    factor(divisor);
	    factor(n/divisor);
	}
	vi primefact( int num )
	{
		v.clear();
		if( num == 1 ) return v;
		q = num;
		factor( num );
		sort( all(v) );
		if( v.empty() ) // primos o 1 
			v.pb( num );
		return v;
	}
}obj;


struct Pollard_Rho
{
	ull q;
	vull v;	
	Pollard_Rho(){}
	Pollard_Rho( ull x ) {
		q = x;
	}
	ull gcd(ull a, ull b){
	    if(b == 0) return a;
	    return gcd( b , a % b );
	}
	ull mul(ull a,ull b,ull c){
	    ull x = 0, y = a % c;
	    while(b > 0){
	        if(b%2 == 1){
	            x = (x+y)%c;
	        }
	        y = (y*2)%c;
	        b /= 2;
	    }
	    return x%c;
	}
	ull modd(ull a,ull b,ull c){
	    ull x=1,y=a; 
	    while(b > 0){
	        if(b%2 == 1){
	            x=mul(x,y,c);
	        }
	        y = mul(y,y,c); 
	        b /= 2;
	    }
	    return x%c;
	}
	bool Miller(ull p,int iteration){
	    if(p<2){
	        return false;
	    }
	    if(p!=2 && p%2==0){
	        return false;
	    }
	    ull s=p-1;
	    while(s%2==0){
	        s/=2;
	    }
	    for(int i=0;i<iteration;i++){
	        ull a=rand()%(p-1)+1,temp=s;
	        ull mod=modd(a,temp,p);
	        while(temp!=p-1 && mod!=1 && mod!=p-1){
	            mod=mul(mod,mod,p);
	            temp *= 2;
	        }
	        if(mod!=p-1 && temp%2==0){
	            return false;
	        }
	    }
	    return true;
	}
	ull rho(ull n){
	    if( n % 2 == 0 ) return 2;
	    ull x = 2 , y = 2 , d = 1;
	    int c = rand() % n + 1;
	    while( d == 1 ){
	        x = (mul( x , x , n ) + c)%n;
	        y = (mul( y , y , n ) + c)%n;
	        y = (mul( y , y , n ) + c)%n;
	        if( x - y >= 0 ) d = gcd( x - y , n );
	        else d = gcd( y - x , n );
	    }
	    return d;
	}
	void factor(ull n){
	    if (n == 1) return;
	    if( Miller(n , 10) ){
	        if(q != n) v.push_back(n);
	        return;
	    }
	    ull divisor = rho(n);
	    factor(divisor);
	    factor(n/divisor);
	}
	vull primefact( ull num )
	{
		v.clear();
		q = num;
		factor( num );
		sort( all(v) );
		if( v.empty() ) // primos o 1 
			v.pb( num );
		return v;
	}
};
################ Teorema chino del resto ###########################################

###########################################################
Teorema chino del resto
-----------------------

Dados k enteros positivos {ni}, tales que ni y nj son coprimos (i!=j).
Para cualquier {ai}, existe x tal que:

x % ni = ai

Todas las soluciones son congruentes modulo N = n1*n2*...*nk

r*ni + s*N/ni = 1 -> ei = s*N/ni   -> ei % nj = 0
                     r*ni + ei = 1 -> ei % ni = 1

x = a1*e1 + a2*e2 + ... + ak*ek

#-----------------------------------------------------------------#
struct EuclidReturn{
    int u,v,d;
  
    EuclidReturn(int _u, int _v, int _d){
        u = _u; v = _v; d = _d;
    }
};
    
EuclidReturn Extended_Euclid(int a, int b){
    if(b==0) return EuclidReturn(1,0,a);
    EuclidReturn aux = Extended_Euclid(b,a%b);
    int v = aux.u-(a/b)*aux.v;
    return EuclidReturn(aux.v,v,aux.d);
}

// ax = b (mod n)
int solveMod(int a,int b,int n){
    EuclidReturn aux = Extended_Euclid(a,n);
    if(b%aux.d==0) return ((aux.u * (b/aux.d))%n+n)%n;
    return -1;// no hay solucuion
}

// ax = 1(mod n)
int modular_inverse(int a, int n){
    EuclidReturn aux = Extended_Euclid(a,n);
    return ((aux.u/aux.d)%n+n)%n;
}
// rem y mod tienen el mismo numero de elementos
long long chinese_remainder(vector<int> rem, vector<int> mod){
    long long ans = rem[0],m = mod[0];
    int n = rem.size();
  
    for(int i=1;i<n;++i){
        int a = modular_inverse(m,mod[i]);
        int b = modular_inverse(mod[i],m);
        ans = (ans*b*mod[i]+rem[i]*a*m)%(m*mod[i]);
        m *= mod[i];
    }
  
    return ans;
}
#-----------------------------------------------------------------#
################## FFT fast fourier transformation O( n log n ) #########################################
// multiply two polynomials
//CDC_MOREFB
#define MOD 99991LL

typedef long double ld;
typedef vector< ld > vld;
typedef vector< vld > vvld;
typedef long long ll;
typedef pair< int , int > pii;
typedef vector< int > vi;
typedef vector< vi > vvi;

ld PI = acos( (ld)(-1.0) );
ll pow( ll a , ll b , ll c ){
	ll ans = 1;
	while( b ){
		if( b & 1 ) ans = (ans * a)%c;
		a = (a * a)%c;
		b >>= 1;
	}
	return ans;
}
ll mod_inv( ll a , ll p ){ return pow(a , p - 2 , p);}

typedef complex<ld> base;
 
void fix( base &x ){
	if(abs(x.imag()) < 1e-16 ){
		x = base( (((ll)round(x.real()))%MOD + MOD)%MOD , 0);
	}
}
void fft (vector<base> & a, bool invert) {
	int n = (int) a.size();
 
	for (int i=1, j=0; i<n; ++i) {
		int bit = n >> 1;
		for (; j>=bit; bit>>=1)
			j -= bit;
		j += bit;
		if (i < j)
			swap (a[i], a[j]);
	}
 
	for (int len=2; len<=n; len<<=1) {
		ld ang = 2.0 * PI /len * (invert ? -1 : 1);
		base wlen (cos(ang), sin(ang));
		for (int i=0; i<n; i+=len) {
			base w (1);
			for (int j=0; j<len/2; ++j) {
				base u = a[i+j],  v = a[i+j+len/2] * w;
				a[i+j] = u + v;
				a[i+j+len/2] = u - v;
				w *= wlen;
			}
		}
	}
	if (invert)
		for (int i=0; i<n; ++i)
			a[i] /= n;
}
void multiply (const vector<ld> & a, const vector<ld> & b, vector<ld> & res) {
	vector<base> fa (a.begin(), a.end()),  fb (b.begin(), b.end());
	size_t n = 1;
	while (n < max (a.size(), b.size()))  n <<= 1;
	n <<= 1;
	fa.resize (n),  fb.resize (n);
 
	fft (fa, false),  fft (fb, false);
	for (size_t i=0; i<n; ++i)
		fa[i] *= fb[i];
	
	fft (fa, true);
 
	res.resize (n);
	for (size_t i=0; i<n; ++i){
		res[i] = (((ll)round( fa[i].real() ))%MOD + MOD)%MOD;
	}
}

void impr( vi &x ){
	REP( i , SZ(x) ) printf( "%d%c" , x[ i ] , (i + 1 == SZ(x)) ? 10 : 32 );
}

vld rec( vvld &T , int lo , int hi ){
	if( lo == hi ) return T[ lo ];
	int mid = (lo + hi) >> 1;
	vld L = rec( T , lo , mid );
	vld R = rec( T , mid + 1 , hi );
	vld X;
	multiply( L , R , X );
	return X;
}
ll solve( ll base , vi &x , int n , int k ){
	// p( x ) = (x + base^v[0]) * ( x + base ^ v[1] ) ....
	vvld T( n );
	REP( i , n )
		T[ i ] = { (ld)pow(base , x[ i ] , MOD) , (ld)1.0 };

	vld v = rec( T , 0 , n - 1 ); 
	ld target = v[ n - k ];

	ll num = (((ll)round( target ))%MOD + MOD)%MOD;
	return num;
}

int main(){
	ll A = 55048LL , B = 44944LL , C = 22019LL;
	//f( n ) = C(A^n - B^n)
	int n , K;
	while( sc( n ) == 1 ){
		sc( K );
		vi x( n );
		REP( i , n ) sc( x[ i ] );
		
		ll SA = solve( A , x , n , K );
		ll SB = solve( B , x , n , K );
		printf( "%lld\n" , (C * (SA - SB + MOD)%MOD)%MOD );
	}
}


#-----------------------------------------------------------------#
################## Karatsuba O( n ^ 1.585 ) #########################################
// Karatsuba 
// multiplicar polinomios
#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )

#define N 100005

typedef int ll;
typedef vector< ll > vll;

void karatsuba( ll *aa, int len1, ll *bb, int len2, ll c[]) {
    if (len1 == 1) {
        REP( i , len2 ) c[ i ] = aa[ 0 ] * bb[ i ];
    }
    else {
        int deg = len1 >> 1;
        int m = len1 + len2 - deg - deg-1;
        ll A[ m ], B[ m ], C[ m ];
         
        fill( A , A + m , 0 );
        karatsuba( aa , deg , bb , deg , A );
         
        int m1 = len1 - deg;
        ll s1[ m1 ]; 
        REP( i , m1 ) s1[ i ] = ( i < deg ? aa[ i ] : 0 ) + aa[ i + deg ];
        int m2 = len2 - deg;
        ll s2[ m2 ];
        REP( i , m2 ) s2[ i ] = ( i < deg ? bb[ i ] : 0 ) + bb[ i + deg ];
         
        fill( C , C + m , 0 );
        karatsuba( aa + deg , m1 , bb + deg , m2 , C );
        fill( B , B + m , 0);
        karatsuba( s1 , m1 , s2 , m2 , B );
         
        REP( i , m ) {
            c[ i ] += A[ i ];
            c[ i + deg ] += B[ i ] - A[ i ] - C[ i ];
            c[ i + deg + deg ] += C[ i ];
        }
    }
}

#-----------------------------------------------------------------#

###################################################################
########################## DP #################################
###################################################################

########################## N - Tiling #################################
//http://www.artofproblemsolving.com/Forum/viewtopic.php?f=151&t=395148
// Para n = 4:
// TJU 3011.   Tiling a Grid With Dominoes
F(n) = F(n-1) + 5*F(n-2) + F(n-3) - F(n-4)
f(n)=0 n<0
f(0)=1 Trivial
f(1)=1
f(2)=5
f(3)=11

//  Para n = 3:
// TJU 1779.   Tri Tiling
F(n)=4*F(n-2)-F(n-4)
// Para n = 2
//UVA 900 - Brick Wall Patterns
Fibonacci

#-----------------------------------------------------------------#
########################## Longest Common Subsequence ( LCS ) #################################
//http://en.wikipedia.org/wiki/Longest_common_subsequence_problem
//UVA 10405 - Longest Common Subsequence
dp( pos1 , pos2 )
	if( pos1 == n1 )return 0;
	if( pos2 == n2 )return 0;
	if( s1[pos1] == s2[pos2] )
		dev = max( 1 + dp( pos1 + 1 , pos2 +1 ) , max( dp( pos1 + 1 , pos2 ) , dp( pos1 , pos2 + 1 ) ) );
	else dev = max( dp( pos1 + 1 , pos2 ) , dp( pos1 , pos2 + 1 ) );
dp( 0 , 0 );
#-----------------------------------------------------------------#
########################## Coin Change #################################
//find the total number of DIFFERENT ways of making changes for any amount of money in cents
//UVA 674 - Coin Change
#define N 10005
#define nV 6
int memo[ N ][ nV ];
int n = 5;
int V[] = { 1 , 5 , 10 , 25 , 50 };
dp( total ,  k )
	if( total == 0 ) return 1;
	if( k == n )return 0;
	dev = dp( total , k + 1 );
	if( total - V[ k ] >= 0 )
		dev += dp( total - V[ k ] , k );
dp( money , 0 )
#-----------------------------------------------------------------#
########################## Edit Distance #################################
/* You are given two strings, A and B. Answer, what is the smallest number of operations you need to transform A to B?
Operations are:
 1) Delete one letter from one of strings
 2) Insert one letter into one of strings
 3) Replace one of letters from one of strings with another letter */
//SPOJ 6219. Edit distance
f( pos1 , pos2 )
    if( pos1 == n1 ) return n2 - pos2;
    if( pos2 == n2 ) return n1 - pos1;
    	dev = min ( 1 + f( pos1 + 1 , pos2 + 1 ), min( 1 + f( pos1 , pos2 + 1 ) , 1 + f( pos1 + 1 , pos2 ) ) );
    if( s1[ pos1 ] == s2[ pos2 ] ) dev = min( dev , f( pos1 + 1 , pos2 + 1 ) );
f( 0 , 0 )
#-----------------------------------------------------------------#
########################## Max 1D Range Sum #################################
int memo[ N ];
// dp( i ) : valuef of maximun sub array [ 0 - i ] and necesarly ends in i
//UVA 10684 - The jackpot
dp( pos )
	if( pos == 0 ) return A[ 0 ];
	dev = max( dp( pos - 1 ) + A[ pos ] , A[ pos ] );
////
////////////////////// falta extender a mas dimensiones :D , abajo esta el hint :D
int n;int a[MAXN];int dp[MAXN];int M[MAXN][MAXN];
int solve()
{
    int ans = dp[0] = a[0];
    for( int i = 1 ; i < n ; ++i )
    {
         dp[i] = max( a[i],dp[i-1] + a[i] );
         ans = max(ans,dp[i]);     
    }    
    return ans;
}
//3D
	int maxi = INT_MIN;
	for( int i1 = 0 ; i1 < n ; ++i1 )
	{
		memset(a,0,sizeof(a));
		for( int i2 = i1 ; i2 < n ; ++i2 )
		{
			for( int j = 0 ; j < n ; ++j )
				a[j]+= M[i2][j];
			maxi = max(maxi,solve());
		}
	}
#-----------------------------------------------------------------#
########################## Table additive 2D( Cuentita ) #################################
//2D
//RECTQUER_CDC
//DP( i , j ) sum of elements of rectangle "A"  [ 1 , i ] [ 1 , j ] 
	if( i == 0 || j == 0 ) return 0;
		DP[ i ][ j ] = DP[ i ][ j - 1 ] + DP[ i - 1 ][ j ] - DP[ i - 1 ][ j - 1 ] + A[ i ][ j ];
// subrectangle 1 - based
	return 	T[ hix ][ hiy ] - T[ lox - 1 ][ hiy ] - T[ hix ][ loy - 1 ] + T[ lox - 1 ][ loy - 1 ];
	
// 3D
//CUBE_CDC
//DP( i , j , k ) sum of elements of rectangle "A"  [ 1 , i ] [ 1 , j ] [ 1 , k ]
		T[ i ][ j ][ k ] = A[ i ][ j ][ k ] + T[ i ][ j ][ k - 1 ] + T[ i ][ j - 1 ][ k ] + T[ i - 1 ][ j ][ k ]
		- T[ i -  1 ][ j - 1 ][ k ] - T[ i - 1 ][ j ][ k - 1 ] - T[ i ][ j - 1 ][ k - 1 ] + T[ i - 1 ][ j - 1 ][ k - 1 ]
// subcube 1 - based
int sum = T[ xhi ][ yhi ][ zhi ] - T[ xlo - 1 ][ yhi ][ zhi ] - T[ xhi ][ ylo - 1 ][ zhi ] - T[ xhi ][ yhi ][ zlo - 1 ]
			+ T[ xlo - 1 ][ ylo - 1 ][ zhi ] + T[ xlo - 1 ][ yhi ][ zlo - 1 ] + T[ xhi ][ ylo - 1 ][ zlo - 1 ] - T[ xlo - 1 ][ ylo - 1 ][ zlo - 1 ];
// falta generalizado para n dimensiones :D			
#-----------------------------------------------------------------#


########################## Subset Sum #################################

//UVA 562 - Dividing coins
#define N 105
#define MAXVAL 505

int n;
int V[ N ];
/*int memo[ N*MAXVAL+ 1 ][ N ];
int dp( int value , int pos )// quiero saber si value se puede formar usando algun subconjunto de vpos ..... vn-1
{
	if( value == 0 ) return 1;
	if( pos == n ) return 0;
	int &dev = memo[ value ][ pos ];
	if( dev == -1 )
	{
		dev = dp( value , pos + 1 );
		if( value - V[ pos ]  >= 0 )
			dev |= dp( value - V[ pos ] , pos + 1 );
	}
	return dev;
}*/
//int DP[ N*MAXVAL+ 1 ][ N ];
//int DP[ N*MAXVAL+ 1 ][ 2 ];
//bool DP[ N*MAXVAL+ 1 ];
//int sum = accumulate( V , V + n , 0 );
//clr( memo , -1 );
/*REP( value , sum + 1 ) DP[ value ][ n ] = 0;
REP( pos , n + 1 ) DP[ 0 ][ pos ] = 1;

for( int pos = n - 1 ; pos >= 0 ; pos-- )
	for( int value = 0 ; value <= sum ; ++value )
	{
		int &dev = DP[ value ][ pos ] = DP[ value ][ pos + 1 ];
		if( value - V[ pos ]  >= 0 )
			dev |= DP[ value - V[ pos ] ][ pos + 1 ];
	}
REP( value , sum + 1 ) DP[ value ][ n&1 ] = 0;
DP[ 0 ][ 0 ] = DP[ 0 ][ 1 ] = 1;

for( int pos = n - 1 ; pos >= 0 ; pos-- )
	for( int value = 0 ; value <= sum ; ++value )
	{
		int &dev = DP[ value ][ pos&1 ] = DP[ value ][ (pos + 1)&1 ];
		if( value - V[ pos ]  >= 0 )
			dev |= DP[ value - V[ pos ] ][ (pos + 1)&1 ];
	}*/
REP( value , sum + 1 ) DP[ value ] = 0;
DP[ 0 ] = 1;

for( int pos = n - 1 ; pos >= 0 ; pos-- )
	for( int value = sum - V[ pos ] ; value >= 0 ; --value )
		DP[ value + V[ pos ] ] |= DP[ value ];
----------------------------------------------------------#
########################## DP over digits( dp peligro o bool peligro ) #################################
//LUCKY2 CDCHEF		DP PELIGRO
//2054 COJ		DP PELIGRO
//9C CDF DIV2		DP PELIGRO
//150B_CDF_DIV2		DP PELIGRO
//1953 PKU		DP PELIGRO
//5295 SPOJ		DP PELIGRO
//http://projecteuler.net/problem=172		DP peligro
// UVA 11361 - Investigating Div-Sum Property
//COJ 2054
string target;
string v,D,H;
int n;
long long b;
long long memo[100][2][2];
long long dp(int pos , int menor ,int started)
{
        if( pos == n ) return 1LL;
        if(  memo[pos][menor][started] !=-1 ) return memo[pos][menor][started];
        long long dev = 0;
        if( started == 0 )
        		dev+=dp(pos+1,1,0);
        if( menor )
        {
                for( int k = 1 - started ; k < b ; ++k )
                        if( v[k] == 'S' )
                                dev += dp(pos+1,1,1);
        }
        else
        {
                int tam = target[pos] - '0';
                for( int k = 1 - started  ; k < tam ; ++k )
                        if( v[k] == 'S' )
                                dev+= dp(pos+1,1,1);  
                if( started == 0 && tam == 0 && v[tam] == 'S')
                	return memo[pos][menor][started] = dev;
                if( v[tam] == 'S' )
                        dev += dp(pos+1,0,1);
        }
        return memo[pos][menor][started] = dev;
}
long long toi(string s){istringstream is(s);long long x;is>>x;return x;}
string tos(long long t){stringstream st; st<<t;return st.str();}
long long f(string s )
{
        target = s;
        n = target.size();
        memset(memo,-1,sizeof(memo));
//      cout<<"ans: "<<dp(0,0)<<endl;   
        return dp(0,0,0); 
}
string prv(string s)
{
        long long cte = toi(s);
        cte--;
        return tos(cte);
}
string to_base(string s)
{
        long long num = toi(s);
        if( num == 0 )return "0";
        string t = "",q;
        while( num > 0 )
        {
                q = '0'+num%b;
                t = q + t;
                num/=b;
        }
        return t;
}
long long g( string s, string t)
{
        string lo = to_base(prv(s));
        string hi = to_base(t);
        //cout<<lo<<" "<<hi<<endl;
        return f(hi) - f(lo);
}
int main()
{
        while( cin>>D>>H>>b>>v )
        {
                if( D=="-1" && H== "-1" && b==-1 && v== "*" ) break;
                cout<< g(D,H) <<endl ;          
        }
}
/// dp with digits + mask in base 3
#include <iostream>
#include <vector>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <algorithm>
using namespace std;

#define REP( i , n ) for( int i = 0 ; i < n ; i++ )
#define clr( t , val ) memset( t , val , sizeof(t) )

#define N 4500005

typedef long long ll;
typedef vector< char > vi;

//map< ll , ll > memo;
ll memo[ N ];
bool done[ N ];
char n;
vi target ;
char dx[] = { 18 , 2 , 2 , 3 , 3, 3, 3, 3, 3,3,3,3,3,3,3,3,3,3,3 };
inline int hash( vi &v ){
    int ans = 0;
    int nv = v.size();
    REP( i , nv ) ans = ans * dx[ i ] + v[ i ];
    return ans;
}
char temp , temp2;
ll dp( vi &v ){
    char pos = v[ 0 ] ;
    bool menor = v[ 1 ] , taked = v[ 2 ];
    if( pos == n ){
        //cout << "concha" << endl;
        return taked;
    }
    ll HASH = hash( v );
    //if( memo[ hash( v ) ] == -1 ) return memo[ HASH ];
    /*cout << "pos : " << v[ 0 ] << endl;
    cout << "menor : " << v[ 1 ] << endl;
    cout << "taked : " << v[ 2 ] << endl;*/
    ll &dev = memo[ HASH ];
    char i;
    if( !done[ HASH ] ){
        done[ HASH ] = 1;
        dev = 0;
        if( taked == 0 ){
            temp = v[ 1 ];
            v[ 0 ]++;v[ 1 ] = 1;
            dev += dp( v );
            v[ 1 ] = temp;
            v[ 0 ]--;
            if( menor == 0 )
            {
                
                for( i = 1 ; i <= target[ pos ] ; ++i )
                {
                    v[ 0 ]++;
                    temp = v[ 1 ];
                    temp2 = v[ 2 ];
                    if( target[ pos ] == i ) v[ 1 ] = 0;
                    else v[ 1 ] = 1;
                    v[ 2 ] = 1;
                    v[ 3 + i ] ++;
                    dev += dp( v );
                    v[ 3 +  i ]--;
                    v[ 0 ]--;
                    v[ 1 ] = temp;
                    v[ 2 ] = temp2;
                }
            }
            else
            {
                for( i = 1 ; i <= 9 ; ++i )
                {
                    v[ 0 ]++;
                    temp = v[ 1 ];
                    v[ 1 ] = 1;
                    temp2 = v[ 2 ];
                    v[ 2 ] = 1;
                    v[ 3 + i ] ++;
                    dev += dp( v );
                    v[ 0 ]--;
                    v[ 1 ] = temp;
                    v[ 2 ] = temp2;
                    v[ 3 + i ]--;
                }
            }
        }
        else
        {
            if( menor == 0 )
            {
                for( i = 0 ; i <= target[ pos ] ; ++i )
                    if( v[ i + 3 ] <= 1 ){
                        v[ 0 ] ++;
                        temp = v[ 1 ];
                        if( target[ pos ] == i ) v[ 1 ] = 0;
                        else v[ 1 ] = 1;
                        v[ i + 3 ]++;
                        dev += dp( v );
                        v[ 0 ] --;
                        v[ 1 ] = temp;
                        v[ i + 3 ]--;
                    }
            }
            else
            {
                for( i = 0 ; i <= 9 ; ++i )
                    if( v[ i + 3 ] <= 1 )
                    {
                        v[ 0 ] ++;
                        v[ i + 3 ] ++;
                        dev += dp( v );
                        v[ 0 ] --;
                        v[ i + 3 ] --;
                    }
            }
        }
    }
    return dev;
}
bool ok;
ll f( ll X ){
    if( X == 0 ) return 0;
    n = 0;
    target.clear();
    while( X )
    {
        target.push_back( X%10 );
        X /= 10;
        n++;    
    }
    reverse( target.begin() , target.end() );
    //REP( i , n ) cout << (int)target[ i ] << char(i + 1 == n ? 10 : 32 );
    //cout << int(n) << endl;
    if( ok ) clr( done , 0 );
    else ok = 1;
    //REP( i , n ) cout << target[ i ] << char( i + 1 == n ? 10 : 32 );
    vi v = vi( 13 );
    return dp( v );
}
int main()
{
    freopen("exchange.in", "r", stdin);freopen("exchange.out", "w", stdout);
    ll L , R; 
    ok = 0;
    //cout << f( 1000 ) << endl;
    while( cin >> L >> R ){
        ll A = f( L - 1 ) , B = f( R );
        //cout << A << " " << B << endl;
        cout << B - A << '\n';
    }
}
################### Longest increasing subsequence ( LIS ) ############################################
// + Reconstruction
//http://en.wikipedia.org/wiki/Longest_increasing_subsequence
// entender :'(
//111_UVA
// O( n^2 ) time , memory
int n;
int LCS( vi &v ){
	v.insert( v.begin() , -1 );
	n++;
	vvi DP( n + 1 , vi( n + 1 ) );
	for( int pos = n - 1 ; pos >= 0 ; --pos )
		for( int last = pos ; last >= 0 ; last -- )
		{
			int &dev = DP[ pos ][ last ] = DP[ pos + 1 ][ last ];
			if( v[ pos ] > v[ last ] ) dev = max( dev , 1 + DP[ pos + 1 ][ pos ] );
		}
	n--;
	return DP[ 0 ][ 0 ];
}
// O( n^2 ) time , O( n ) memory
int LIS( vi &v ){
	vi DP( n );
	DP[ 0 ] = 1;
	//REP( i , n ) cout << v[ i ] << char( i + 1 == n ? 10 : 32 );
	for( int i = 1 ; i < n ; ++i ){
		int &dev = DP[ i ] = 1;
		for( int j = 0 ; j < i ; ++j )
			if( v[ j ] < v[ i ] ) dev = max( dev , 1 + DP[ j ] );
	}
	return *max_element( all( DP ) );
}
//
vector<int> LIS(vector<int> X){
    int n = X.size(),L = 0,M[n+1],P[n]; 
    int lo,hi,mi;    
    L = 0;
    M[0] = 0;
    
    for(int i=0,j;i<n;i++){
        lo = 0; hi = L;
        
        while(lo!=hi){
            mi = (lo+hi+1)/2;
            
            if(X[M[mi]]<X[i]) lo = mi;
            else hi = mi-1;
        }
        
        j = lo;      
        P[i] = M[j];
        
        if(j==L || X[i]<X[M[j+1]]){
            M[j+1] = i;
            L = max(L,j+1);
        }
    }
    
    int a[L];
    for(int i=L-1,j=M[L];i>=0;i--){
        a[i] = X[j];
        j = P[j];
    }
    
    return vector<int>(a,a+L);
}
// O( nlogn ) time , O( n ) memory
// by Chen

int LIS( vi &a ){
	vi b;
	for( auto x : a ){
		int j = upper_bound( all( b ) , x ) - b.begin();
		// (lower) a < b < c 
		// (upper) a <= b <= c 
		if( j == SZ(b) ) b.pb( x );
		else b[ j ] = x;
	}
	return SZ(b);
}
vi lis( vi &v ){
	if( !SZ(v) ) return vi();
    int n = SZ( v );
    vi sig( n , -1 ) , id( n ) , r( n );
    vi mini;
    REP( i , n ){
        int j = lower_bound( all( mini ) , v[ i ] ) - mini.begin();
        if( j == SZ( mini ) ) mini.pb( 0 );
        mini[ j ] = v[ i ];
        id[ j ] = i;
        if( j ) sig[ i ] = id[ j - 1 ];
    }
    int from = id[ SZ( mini ) - 1 ];
    int top = SZ( mini );
    do{
        r[--top] = from;
        from = sig[ from ];
    }while( from != -1 );
    vi vec;
    REP( i , SZ( mini ) ) vec.pb( v[ r[ i ] ] );//creo que r[ i ] es el indice de v
    return vec;
}
///LIS2D if we need x1 < x2 && y1 < y2 sort as pairs with( x , -y )

//XMEN SPOJ
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define N 100000

typedef vector< int > vi;
typedef long long ll;

int mapa[ N + 5 ];
int main(){
	int cases , n , x ;
	sc( cases );
	REP( tc , cases ){
		sc( n );
		REP( i , n ){
			sc( x );
			x --;
			mapa[ x ] = i;
		}
		vi b;
		REP( i , n ){
			sc( x );
			x --;
			x = mapa[ x ];
			int pos = lower_bound( all( b ) , x ) - b.begin();
			if( pos == SZ( b ) ) b.pb( x );
			else b[ pos ] = x;
		}
		printf( "%d\n" , SZ( b ) );
	}
}

#-----------------------------------------------------------------#
//VALUE EXPECTED
//F(x) = p1*( c1 + f(x') ) + p2*( c2 + f(x'') )

//Secretary_problem https://www.youtube.com/watch?v=ZWib5olGbQ0
//C:\luis\GYM\100324\I.cpp

int main(){
	int n;
	while( cin >> n ){
		if( n == 1 ){
			puts( "1" );
			continue;
		}
		double ans = 0;
		for( int r = 1 ; r <= n ; ++r ){
			double fact = 0;
			for( int i = r ; i < n ; ++i ) fact += 1.0 / (double)(i);
			ans = max( ans , fact * r / (double) n );
		}
		printf( "%.10f\n" , ans );
	}
}
################# DP in DAG #######################################
// longestPathInDag
// with cycle finding   220C_CDF_DIV2
bool cycle;
int V;
vi E[ N*N ];//-1 = unseen, -2 = processing, nonnegative = computed value
int dp[ N*N ]; 
bool good[ N*N ];

void dfs( int u ){
	if( cycle ) return;
	if( dp[ u ] == -2 ){
		cycle = 1;
		return;
	}
	if( dp[ u ] != -1 ) return;
	dp[ u ] = -2;
	int best = 0;
	FOR( v , E[ u ] ){
		dfs( *v );
		best = max( best , 1 + dp[ *v ] );
	}
	dp[ u ] = best;
}
int longestPathInDag()
{
	cycle = 0;
	clr( dp , -1 );
	REP( i , V ) if( good[ i ] ) dfs( i );
	if( cycle ) return -1;
	int best = 0;
	REP( i , V ) best = max( best , dp[ i ] );
	return best;
}

###################################################################
################# GRAPHS #######################################
###################################################################


######################## UNION FIND ###################################

REP( i , n ) id[i] = i ; // init
int Find( int x ){ return id[ x ] = (id[ x ] == x ? x : Find( id[ x ] ) );}
void Union(int x, int y) {id[Find(x)] = Find(y);} // Find( x ) != Find( y )
//  with path compression
REP( i , SZ ) parent[i] = i , rank[i] = 1;
void Union(int a, int b)
{
	int pa = Find(a);
	int pb = Find(b);
	if(pa != pb)
	{
		if(rank[pa] < rank[pb]) parent[pa] = pb;
		else if(rank[pa] > rank[pb]) parent[pb] = pa;
		else
		{
			parent[pb] = pa;
			rank[pa]++;
		}
	}
}
// Estructure edge for Kruskal Algo
// ACM 2515 - Networking 
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define N 50

typedef vector< int > vi;
typedef long long ll;

int id[ N + 5 ];
int Find( int x ){ return id[ x ] = (id[ x ] == x ? x : Find( id[ x ] ) );}
struct Edge{
	int u , v , w;
	Edge(){}
	Edge( int u , int v , int w ) : u( u ) , v( v ) , w( w ) {}
};
bool operator < ( const Edge &a , const Edge &b ){ return a.w < b.w ;}
int main(){
	int n , m , u , v , w;
	while( sc( n ) == 1 ){
		if( !n ) break;
		sc( m );
		REP( i , N ) id[ i ] = i;
		vector< Edge > E;
		REP( i , m ){
			sc( u ) , sc( v ) , sc( w );
			u -- , v --;
			E.pb( Edge( u , v , w ) );
		}
		sort( all( E ) );
		int ans = 0;
		REP( i , SZ( E ) ){
			int pu = Find( E[ i ].u ) , pv = Find( E[ i ].v );
			if( pu != pv ){
				ans += E[ i ].w;
				id[ pu ] = pv;
			}
		}
		printf( "%d\n" , ans );
	}
}


/// SOPORTA aristas repetidas porque en este caso ordena pesos y se queda con la optima
#-----------------------------------------------------------------#
########################## DJKSTRA #################################
// ( Optimized )
//Codeforces Alpha Round #20 (Codeforces format) C - Dijkstra?
#include <bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; i++ )
#define clr( t , val ) memset( t , val , sizeof(t) )

#define all(v)  v.begin() , v.end()
#define pb push_back
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define test() cerr << "hola que hace ?" << endl;
#define DEBUG( x ) cerr <<  #x << "=" << x << endl;
#define DEBUG2( x , y ) cerr << #x << "=" << x << " " << #y << "=" << y << endl;

#define N 100000

typedef long long ll;
typedef vector< ll > vll;
typedef pair< int , int > pii;
typedef vector< int > vi;

const ll INF = (1LL << 61);

vi G[ N + 5 ] ;
vll C[ N + 5 ];
int m , n ;
void add( int u , int v , ll w ){
    G[ u ].pb( v );
    C[ u ].pb( w );
}
struct Node{
    int id ;
    ll dist;
    Node( int id , ll dist ) : id( id ) , dist( dist ) {}
};
bool operator < ( const Node &a , const Node &b ){ return a.dist > b.dist; }

vi dijkstra( int s ){
    vll dist( n , INF );
    vi vis( n ) , pa( n , -1 );
    priority_queue< Node > Q;
    dist[ s ] = 0;
    Q.push( Node( s , 0 ) );
    while( !Q.empty() ){
        Node U = Q.top() ; Q.pop();
        int u = U.id ;
        if( vis[ u ] ) continue;
        vis[ u ] = 1;
        REP( i , SZ( G[ u ] ) ){
            int v = G[ u ][ i ];
            ll w = C[ u ][ i ];
            if( !vis[ v ] && dist[ u ] + w < dist[ v ] ){
                dist[ v ] = dist[ u ] + w;
                pa[ v ] = u;
                Q.push( Node( v , dist[ v ] ) );
            }
        }
    }
    if( dist[ n - 1 ] >= INF ) return vi( 1 , -1 );
    int u = n - 1 ;
    vi vec;
    while( 1 ){
        vec.pb( u + 1 );
        if( pa[ u ] == -1 ) break;
        u = pa[ u ];
    }
    reverse( all( vec ) );
    return vec;
}

int main(){
    while( sc( n ) == 1 ){
        sc( m );
        REP( i , n ) G[ i ].clear() , C[ i ].clear();
        REP( i , m ){
            int u , v , w;
            sc( u ) , sc( v ) , sc( w );
            u -- , v --;
            add( u , v , w );
            add( v , u , w );
        }
        vi vec = dijkstra( 0 );
        REP( i , SZ( vec ) ) printf( "%d%c" , vec[ i ] , (i + 1 == SZ(vec)) ? 10 : 32 );
    }
}


#-----------------------------------------------------------------#
########################## Euler Tour #################################
//232E_CDF_DIV2
int timer;
int in[ N + 5 ] , out[ N + 5 ] , depth[ N + 5 ];
void dfs( int u , int d ){
    in[ u ] = timer ++;
    depth[ u ] = d;
    FOR( v , E[ u ] ) dfs( *v , d + 1 );
    out[ u ] = timer ++;
}
########################## Cycle Detection #################################
//Codeforces Round #317 [AimFund Thanks-Round] (Div. 2) E. CNF 2

struct Edge{
	int u , v , x , flag , id;
	Edge(){}
	Edge( int u , int v , int x , int flag , int id ) : u( u ) , v( v ) , x( x ) , flag( flag ) , id( id ) {}
};

vpii bucket[ M + 1 ];
Edge pa[ N + 1 ];
vector< Edge > G[ N + 1 ];
queue< int > Q;
int vis[ N + 1 ] , visited[ N + 1 ];
int var[ M + 1 ];

void update( Edge &e ){
	int x = e.x , flag = e.flag;
	var[ x ] = flag;
}
void bfs(){
	while( !Q.empty() ){
		int u = Q.front(); Q.pop();
		for( auto e : G[ u ] ){
			int v = e.v;
			if( !vis[ v ] ){
				vis[ v ] = 1;
				Q.push( v );
				update( e );
			}
		}
	}
}
bool found;
void dfs( int u , int p = -1 ){
	//DEBUG( u );
	if( found ) return;
	visited[ u ] = 1;
	for( auto e : G[ u ] ){
		if( found ) return;
		int v = e.v , id = e.id;
		if( id == p ) continue;
		if( visited[ v ] ){
			update( e );
			
			Q.push( v );
			vis[ v ] = 1;
			//DEBUG2( u , v );
			while( u != v ){
				
				e = pa[ u ];
				update( e );
				
				Q.push( u );
				vis[ u ] = 1;
				
				u = e.u;
			}
			found = 1;
			return;
		}
		pa[ v ] = e;
		dfs( v , id );
	}
}

int main() {
	int n , m;
	while( sc( n ) == 1 ){
		sc( m );
		REP( i , m ) bucket[ i ].clear();
		REP( i , n ) G[ i ].clear();
		
		REP( u , n ){
			int K;
			sc( K );
			REP( i , K ){
				int x;
				sc( x );
				int X = abs( x );
				X --;
				int sign = ( x > 0 );
				bucket[ X ].pb( mp( u , sign ) );
			}
		}
		
		
		int e = 0;
		REP( i , n ) vis[ i ] = visited[ i ] = 0;
		REP( i , m ) var[ i ] = -1;
		REP( i , m ){
			if( !SZ( bucket[ i ] ) ) continue; 
			if( SZ( bucket[ i ] ) == 1 ){
				int u = bucket[ i ][ 0 ].fi , sign = bucket[ i ][ 0 ].se;
				var[ i ] = sign;
				vis[ u ] = 1;
				Q.push( u );
			}else{
				int u = bucket[ i ][ 0 ].fi , v = bucket[ i ][ 1 ].fi , sign = bucket[ i ][ 0 ].se;
				if( u == v ){
					if( bucket[ i ][ 0 ].se == 0 && bucket[ i ][ 1 ].se == 0 ) var[ i ] = 1;
					else var[ i ] = 0;
					vis[ u ] = 1;
					Q.push( u );
				}else{
					if( bucket[ i ][ 0 ].se == bucket[ i ][ 1 ].se ){
						if( bucket[ i ][ 0 ].se == 0 ) var[ i ] = 0;
						else var[ i ] = 1;
						vis[ u ] = 1;
						Q.push( u );
						vis[ v ] = 1;
						Q.push( v );
					}else{
						G[ u ].pb( Edge( u , v , i , !sign , e ));
						G[ v ].pb( Edge( v , u , i , sign , e ));
						e ++;
					}
				}
			}
		}
		bfs();
		bool ok = 1;
		REP( i , n ){
			if( !vis[ i ] ){
				found = 0;
				//DEBUG( i );
				dfs( i , -1 );
				if( !found ){
					ok = 0;
					break;
				}
				bfs();
			}
		}
		puts( ok ? "YES" : "NO" );
		if( ok ){
			REP( i , m ) if( var[ i ] == -1 ) var[ i ] = 0;
			REP( i , m ) putchar( '0' + var[ i ] );
			puts( "" );
		}
	}
}


#-----------------------------------------------------------------#
########################## LCA #################################
// SPOJ 913. Query on a tree II
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define LOGN 14 
#define N 10000

#define test puts( "******************test**********************" );

typedef vector< int > vi;

int rmq[ LOGN + 1 ][ N + 5 ];

int depth[ N + 5 ] , dist[ N + 5 ];
bool vis[ N + 5 ];
vi G[ N + 5 ] , C[ N + 5 ];
void dfs( int u , int p = -1 ){
	vis[ u ] = 1;
	REP( i , SZ( G[ u ] ) ){
		int v = G[ u ][ i ];
		if( v != p && !vis[ v ] ){
			rmq[ 0 ][ v ] = u;
			depth[ v ] = depth[ u ] + 1;
			dist[ v ] = dist[ u ] + C[ u ][ i ];
			dfs( v , u );
		}
	}
}
void addEdge( int u , int v , int w ){
	G[ u ].pb( v );
	C[ u ].pb( w );
}

int LCA( int a , int b ){
	if( depth[ a ] > depth[ b ] ) swap( a , b );
	int dif = depth[ b ] - depth[ a ];
	for( int i = 0 ; i <= LOGN ; ++i ) if( dif & (1<<i) ) b = rmq[ i ][ b ];
	if( a == b ) return a;
	for( int k = LOGN ; k >= 0 ; --k )
		if( rmq[ k ][ a ] != rmq[ k ][ b ] ) a = rmq[ k ][ a ] , b = rmq[ k ][ b ];
	return rmq[ 0 ][ a ];
}
int findKth( int a , int b , int K , int lca ){
	if( depth[ a ] - depth[ lca ] >= K ){
		for( int i = 0 ; i <= LOGN ; ++i ) if( K & ( 1 << i ) ) a = rmq[ i ][ a ];
		return a;
	}
	
	K = (depth[ a ] + depth[ b ] - (depth[ lca ]<<1)) - K;
	swap( a , b );
	for( int i = 0 ; i <= LOGN ; ++i ) if( K & ( 1 << i ) ) a = rmq[ i ][ a ];
	return a;

}
int main(){
	char op[ 15 ];
	int cases , n , u , v , w , K;
	sc( cases );
	REP( tc , cases ){
		sc( n );
		REP( i , N ) G[ i ].clear() , C[ i ].clear();
		
		REP( i , n - 1 ){
			sc( u ) , sc( v ) , sc( w );
			u -- , v --;
			addEdge( u , v , w );
			addEdge( v , u , w );
		}
		clr( rmq , -1 );
		clr( vis , 0 );
		clr( depth , 0 );//depends of number of trees
		dfs( 0 );//it depends if there is many trees , asuming just one root (root = 0)
		for( int k = 1 ; k <= LOGN ; ++k )
			REP( i , n ) if( rmq[ k - 1 ][ i ] != -1 ) rmq[ k ][ i ] = rmq[ k - 1 ][ rmq[ k - 1 ][ i ] ];
		if( tc ) puts( "" );
		while( 1 ){
			scanf( "%s" , op );
			if( op[ 0 ] == 'K' ){
				sc( u ) , sc( v ) , sc( K );
				u -- , v --;
				printf( "%d\n" , findKth( u , v , K - 1 , LCA( u , v ) ) + 1 );
			}else if( op[ 1 ] == 'I' ){
				sc( u ) , sc( v );
				u -- , v --;
				printf( "%d\n" , dist[ u ] + dist[ v ] - (dist[ LCA( u , v ) ]<<1) );
			}else break;
		}
	}
}
########################## FLOYD - WARSHALL #################################
void init(){
    REP( i , n ) REP( j , n ) d[ i ][ j ] = INF;
    // actualize table of dist "d"
    REP( i , n ) d[ i ][ i ] = 0;	    	
}
void floyd(){
	REP( k , n ) REP( i , n ) REP( j , n )
        d[ i ][ j ] = min( d[ i ][ j ] , d[ i ][ k ] + d[ k ][ j ] );	
}  
#-----------------------------------------------------------------#
########################## TOPOLOGICAL_SORT #################################
//http://ahmed-aly.com/Standings.jsp?ID=2954
//11371_SPOJ
#define MAXN 100

	for( int i = 0 ; i < m ; ++i )
	{
		G[u].push_back(v);
		in[v]++;
	}
	priority_queue <int> Q;
	for( int i = 0 ; i < n ; ++i )
		if( in[i] == 0 )
			Q.push(-i);
	vector< int >orden;
	while( !Q.empty() )
	{
		int u = Q.top();
		u = -u;
		Q.pop();    
		orden.push_back(u);
		int nG = G[u].size();
		for( int i = 0 ; i < nG ; ++i )
		{
			int v = G[u][i];
			in[v]--;
			if( in[v] == 0 )
			Q.push(-v);
		}
	}
}
// recrusivily
void topsort( int u ){
	vis[ u ] = 1;
	FOR( v , dag[ u ] )
		if( !vis[ *v ] ) topsort( *v );
	cola[ sz ++ ] = u;
}
#-----------------------------------------------------------------#
########################## SCC #################################
//SCC of tarjan
// Aplicacion 2SAT 4185_ACM , 3683_PKU , 1794_TJU , 27D_CDF_DIV2
#define N 100
#define M 100000

typedef long long ll;
typedef vector< int > vi;
typedef vector< vi > vvi;

int n , timer , top;
vi G[ N + 5 ];
int dfsn[ N + 5 ] , pila[ N + 5 ] , inpila[ N + 5 ] , comp[ N + 5 ];

int dfs( int u ){
	int low = dfsn[ u ] = ++timer;
	inpila[ pila[ top ++ ] = u ] = 1;
	for( int v : G[ u ] ){
		if( dfsn[ v ] == 0 ) low = min( low , dfs( v ) );
		else if( inpila[ v ] ) low = min( low , dfsn[ v ] );
	}
	if( low == dfsn[ u ] ){
		int fin;
		do{
			fin = pila[ --top ];
			inpila[ fin ] = 0;
			comp[ fin ] = u;
		}while( fin != u );
	}
	return low;
}

void SCC(){
	clr( dfsn , 0 );
	top = timer = 0;
	REP( i , n ) if( !dfsn[ i ] ) dfs( i );
}

//////////////////////////////////////////////2-sat + reco
//100430GYM_A
///Now consider a graph with 2n vertices; For each of (xi ∨ yi) s we add two directed edges
//Consider f=(x1 or y1) and (x2 or y2) and ...  and (xn or yn).
//From \ACxi to yi
//From \ACyi to xi

int n , timer , top;
vi E[ N + 5 ];
int dfsn[ N + 5 ] , pila[ N + 5 ] , inpila[ N + 5 ] , comp[ N + 5 ];

int dfs( int u ){
	int low = dfsn[ u ] = ++timer;
	inpila[ pila[ top ++ ] = u ] = 1;
	for( auto v : E[ u ] )
		if( dfsn[ v ] == 0 ) low = min( low , dfs( v ) );
		else if( inpila[ v ] ) low = min( low , dfsn[ v ] );
	if( low == dfsn[ u ] ){
		int fin;
		do{
			fin = pila[ --top ];
			inpila[ fin ] = 0;
			comp[ fin ] = u;
		}while( fin != u );
	}
	return low;
}

void SCC(){
	REP( i , 2 * n ) dfsn[ i ] = 0;
	timer = 0;
	REP( i , 2 * n ) if( !dfsn[ i ] ) dfs( i );
}
vi dag[ N + 5 ];
int sz ;
int vis[ N + 5 ] , decision[ N + 5 ] , cola[ N + 5 ];

void topsort( int u ){
	vis[ u ] = 1;
	for( auto v : dag[ u ] )
		if( !vis[ v ] ) topsort( v );
	cola[ sz ++ ] = u;
}
void paint( int u ){
	decision[ u ] = 1;
	for( auto v : dag[ u ] )
		if( decision[ v ] == -1 ) paint( v );
}
vvi T;
void solve(){
	SCC();
	REP( i , n ) if( comp[ 2 * i ] == comp[ 2 * i + 1 ] ){
		puts( "NO" );
		return;
	}
	puts( "YES" );
	
	REP( u , 2 * n )for( auto v : E[ u ] ){
		int i = comp[ u ] , j = comp[ v ];
		if( i != j ) dag[ i ].pb( j );
	}
	REP( i , 2 * n ) vis[ i ] = 0;
	sz = 0;
	REP( i , 2 * n ) if( comp[ i ] == i && !vis[ i ] ) topsort( i );
	REP( i , 2 * n ) decision[ i ] = -1;
	reverse( cola , cola + sz );
	REP( i , sz )
		if( decision[ cola[ i ] ] == -1 ){
			decision[ cola[ i ] ] = 0;
			paint( comp[ cola[ i ] ^ 1 ] );
		}
	
	vi ans;
	REP( i , n ) ans.pb( T[ i ][ decision[ comp[ 2 * i ] ] ^ 1 ] );
	REP( i , n ) printf( "%d%c" , ans[ i ] + 1 , (i + 1 == n) ? 10 : 32 );
}
void add( int u , int v ){// u or v
	E[ u ^ 1 ].pb( v );
	E[ v ^ 1 ].pb( u );
}
int main(){
	freopen( "chip.in" , "r" , stdin );
	freopen( "chip.out" , "w" , stdout );
	while( sc( n ) == 1 ){
		vi col( n );
		REP( i , n ) sc( col[ i ] );
		T = vvi( n );
		vi var;
		REP( i , 2 * n ){
			int x;
			sc( x );
			x --;
			T[ x ].pb( i );
			var.pb( (SZ( T[ x ] ) == 1) ? (2 * x) : (2 * x + 1) );
		}
		
		REP( i , 2 * n ) E[ i ].clear() , dag[ i ].clear();
		REP( u , 2 * n ){
			int v = (u + 1)%(2 * n);
			int idu = var[ u ] , idv = var[ v ];
			if( idu & 1 ) idu ^= 1;
			if( idv & 1 ) idv ^= 1;
			idu >>= 1 , idv >>= 1;
			if( col[ idu ] == col[ idv ] ){
				add( var[ u ] ^ 1 , var[ v ] ^ 1 );
			}
		}
		solve();
	}
}

########################## BCC #################################
//2-vertex-connected
// Puentes
//5796_ACM 

#define VIZ(e,x) (orig[e] == x? dest[e] : orig[e])
int n,m,fin;
int orig[M + 1], dest[M + 1], peso[M + 1],pila[M + 1], top = 0;
vi E[N + 1];
int low[N + 1], dfsn[N + 1], part[N + 1], timer;
int ponte[M + 1], bicomp[M + 1] , nbicomp;

int dfsbcc (int u, int p = -1){
	low[u] = dfsn[u] = ++timer;
	int ch = 0;
	for( auto e : E[ u ] ){
		int v = VIZ (e, u);
		if (dfsn[v] == 0){
			pila[top++] = e;
			dfsbcc (v, u);
			low[u] = min (low[u], low[v]);
			ch++;
			if (low[v] >= dfsn[u]){
				part[u] = 1;
				do{
					fin = pila[--top];
					bicomp[fin] = nbicomp;
				}while (fin != e);
				nbicomp++;
			}
			if (low[v] == dfsn[v]) ponte[e] = 1;
		}else if (v!=p && dfsn[v] < dfsn[u]){
			pila[top++] = e;
			low[u] = min (low[u], dfsn[v]);
		}
	}
	return ch;
}
void bcc (){
	REP( i , n ) part[ i ] = dfsn[ i ] = 0;
	REP( i , m ) ponte[ i ] = 0;
	nbicomp = timer = 0;
	REP( i , n ) if (dfsn[ i ] == 0) part[ i ] = dfsbcc( i ) >= 2;
}

int main()
{
	cin >> n >> m;
	f(i,0,m){
		int u,v; cin >> u >> v; u--; v--;
		orig[i] = u;
		dest[i] = v;
		E[u].pb (i);
		E[v].pb (i);
	}
	bcc();
	f(i,0,n) cout << part[i]; cout << endl;
	f(i,0,m) cout << ponte[i]; cout << endl;
	f(i,0,m) cout << bicomp[i]; cout << endl;
}
######################### Bicoloring #################################
/// BFS
vi G[ N ];
int vis[ N ] , COLOR[ N ];
bool bfs( int s )
{
	queue< int > Q;
	Q.push( s );
	COLOR[ s ] = 0;
	vis[ s ] = 1;
	while( !Q.empty() ){
		int u = Q.front();
		Q.pop();
		FOR( v , G[ u ] ){
			if( vis[ *v ] ){
				if( COLOR[ *v ] == COLOR[ u ] ) return 0;
			}
			else{
				COLOR[ *v ] = COLOR[ u ]^1;
				vis[ *v ] = 1;
				Q.push( *v );
			}
		}
	}
	return 1;
}
######################### Heavy Light Decomposition HLD #################################
//1553_TI
#define FOR(it,A) for(typeof A.begin() it = A.begin(); it!=A.end(); it++)
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define all( v ) v.begin() , v.end()
#define pb push_back

#define N 100005

#define v1 ( (node<<1) + 1 )
#define v2 v1 + 1
#define med ( ( a + b )>>1 )
#define LEFT v1 , a , med
#define RIGHT v2 , med + 1 , b
#define JOIN T[ node ] = max( T[ v1 ] , T[ v2 ] )

typedef vector< int > vi;

struct stripe {
	vi T;
	int len;
	stripe() {}
	stripe( vi &a ) : len( a.size() ) {
		int off = 1;
		for (; off < len; off<<=1);
		T = vi( off<<1 );
	}
	void update( int pos , int val ) { update( pos , val , 0 , 0 , len - 1 ); }
	void update( int pos , int val , int node , int a , int b ) {
		if ( pos > b || a > pos ) return;
		if ( a == b ) {
			T[ node ] += val;
			return;
		}
		update( pos , val , LEFT );
		update( pos , val , RIGHT );
		JOIN;
	}
	int get2( int lo , int hi ) { return get2( lo , hi , 0 , 0 , len - 1 ); }
	int get2( int lo , int hi , int node , int a , int b ) {
		if( lo > b || a > hi ) return 0;
		if( a >= lo && b <= hi ) return T[ node ];
		return max( get2( lo , hi , LEFT ) ,  get2( lo , hi , RIGHT ) );
	}

} st[ N ];

int chain[ N ] , tam[ N ] , h[ N ] , p[ N ];

int n , q ; 
int peso[ N ];
int csz , cola[ N ];
vi E[ N ];

int querynodos( int u , int v ) {
	int maxi = -1;
	while( chain[ u ] != chain[ v ] ) {
		if( h[ chain[ u ]]  < h[ chain[ v ] ]) swap( u , v );
		int c = chain[ u ] , len = st[ c ].len;
		maxi = max( maxi , st[ c ].get2( 0 , h[ u ] - h[ c ] ) );
		u = p[ chain[ u ] ];
	}
	if( h[ u ] < h[ v ] ) swap( u , v );
	int c = chain[ u ] , len = st[ c ].len;
	maxi = max( maxi , st[ c ].get2( h[ v ] - h[ c ] , h[ u ] - h[ c ] ) );
	return maxi;
}
int main(){
	cin >> n;
	REP( i , n - 1 ) {
		int u,v; 
		scanf( "%d%d" , &u , &v ); u--; v--;
		E[ u ].pb( v );
		E[ v ].pb( u );
	}
	clr( p , -1 );
	csz = 0;
	cola[ csz++ ] = 0;
	p[ 0 ] = 0;
	h[ 0 ] = 0;
	REP( i , csz ) {
		int u = cola[ i ];
		FOR( e , E[ u ] ) {
			int v = *e;
			if ( ~p[ v ] ) continue;
			cola[ csz++ ] = v;
			p[ v ] = u;
			h[ v ] = h[ u ] + 1;
		}
	}
	for( int i = csz - 1 ; i >= 0 ; --i ) {
		int u = cola[ i ];
		tam[ u ] = 1;
		FOR( e , E[ u ] ) {
			int v = *e;
			if( p[ u ] == v ) continue;
			tam[ u ] += tam[ v ];
		}
	}
	clr( chain , -1 );
	REP( i , csz ) {
		int u = cola[ i ];
		if ( ~chain[ u ] ) continue;
		chain[ u ] = u;
		int v = u;
		while( 1 ) { 
			int next = -1;
			FOR( v , E[ u ] ) if( p[ *v ] == u )
				if ( next ==-1 || tam[ next ] < tam[ *v ] ) next = *v;
			if( next == -1 ) break;
			chain[ next ] = chain[ u ];
			u = next;
		}
		int len = h[ u ] - h[ v ] + 1;
		vi a( len );
		/*
		REP( i , len ) {
			a[ i ] = peso[u];
			u = p[ u ];
		}
		reverse( all( a ) );*/
		st[ v ] = stripe(a);
	}
	cin >> q;
	REP( i , q ) {
		char op; 
		cin >> op;
		int u,v;
		if( op == 'I' ) {
			scanf( "%d%d" , &u , &v ); u--;//// increment in v to u
			int c = chain[ u ] , len = st[ c ].len;
			st[ c ].update( h[ u ] - h[ c ] , v );
		}else {
			scanf("%d%d", &u, &v); u-- , v--;
			int ans = querynodos( u , v );
			printf( "%d\n" , ans );
		}
	}
}

/////Codeforces Round #329 (Div. 2) D. Happy Tree Party
typedef ll tipo;

const ll INF = 2000000000000000000LL;
struct data{
	tipo val;
	data(){ val = 1; }
	data( tipo val ) : val( val ) {}
};
data operator + ( const data &a , const data &b ){
	ld x = a.val , y = b.val;
	if( x * y > INF ) return data( INF );
	return data( a.val * b.val );
}
#define mid ((a + b)>>1)

struct Node{
	typedef Node * pNode;
	data val;
	int len;
	Node * l , * r;
	Node(){ val = data() , len = 0; l = r = NULL; }
	Node( tipo val ) : l( NULL ) , r( NULL ) , val( val ) , len( 1 ) {}
	Node( Node * l , Node * r ) : l( l ) , r( r ) , val() , len( 0 ) {
		if( l ){
			val = val + (l->val);
			len = len + (l->len);
		}
		if( r ){
			val = val + (r->val);
			len = len + (r->len);
		}
	}
	
	Node( vector< tipo > &A ){		
		pNode p = build( 0 , SZ(A) - 1 , A );
		*(this) = *p;
	}
	pNode build( int a , int b , vector< tipo > &A ){
		if( a == b ) return new Node( A[ a ] );
		return new Node( build( a , mid , A ) , build( mid + 1 , b , A ) );
	}
	data get( int lo , int hi ) { 
		return get( this , 0 , len - 1 , lo , hi ); 
	}
	data get( pNode t , int a , int b , int lo , int hi ) {
		if( lo > b || a > hi ) return data();
		if( a >= lo && b <= hi ) return (t -> val);
		return get( (t -> l) , a , mid , lo , hi ) + get( (t -> r) , mid + 1 , b , lo , hi );
	}
	void update( int pos , tipo val ){
		update( this , 0 , len - 1 , pos , val );
	}
	void update( pNode t , int a , int b , int pos , tipo val ){
		if( pos > b || a > pos ) return;
		if( pos <= a && b <= pos ){
			(t -> val) = val;
			return;
		}
		update( (t -> l) , a , mid , pos , val );
		update( (t -> r) , mid + 1 , b , pos , val );
		(t -> val) = ((t -> l) -> val) + ((t -> r) -> val);
	}
} st[ N + 5 ];

int chain[ N + 5 ] , tam[ N + 5 ] , h[ N + 5 ] , pa[ N + 5 ];
ll paW[ N + 5 ];
vi G[ N + 5 ];
vll C[ N + 5 ];
data querynodos( int u , int v ) {
	data S;
	while( chain[ u ] != chain[ v ] ) {
		if( h[ chain[ u ] ] < h[ chain[ v ] ] ) swap( u , v );
		int c = chain[ u ];
		S = S + st[ c ].get( 0 , h[ u ] - h[ c ] );
		u = pa[ chain[ u ] ];
	}
	if( h[ u ] < h[ v ] ) swap( u , v );
	int c = chain[ u ];
	S = S + st[ c ].get( h[ v ] - h[ c ] + 1 , h[ u ] - h[ c ] );// h[ v ] - h[ c ] + (1/0) (for edges/for nodes) 
	return S;
}
void impr( int n ){
	REP( u , n ){
		if( !u ) continue;
		DEBUG2( u + 1 , querynodos( u , pa[ u ] ).val );
	}
}
int main(){
	int n , Q;
	while( sc( n ) == 1 ){
		sc( Q );
		REP( i , n ) G[ i ].clear() , C[ i ].clear();
		
		vi orig(n - 1) , dest(n - 1);
		REP( it , n - 1 ){
			int u , v; 
			sc( u ) , sc( v );
			u -- , v --;
			orig[ it ] = u;
			dest[ it ] = v;
			ll w;
			scanf( "%I64d" , &w );

			G[ u ].pb( v );
			C[ u ].pb( w );
			
			G[ v ].pb( u );
			C[ v ].pb( w );
		}

		REP( i , n ) pa[ i ] = -1;
		
		vi cola;
		cola.pb( 0 );
		pa[ 0 ] = -2;
		paW[ 0 ] = -1;
		h[ 0 ] = 0;
		REP( i , SZ( cola ) ){
			int u = cola[ i ];
			REP( i , SZ( G[ u ] ) ){
				int v = G[ u ][ i ];
				ll w = C[ u ][ i ];
				if( pa[ v ] != -1 ) continue;
				cola.pb( v );
				pa[ v ] = u;
				paW[ v ] = w;
				h[ v ] = h[ u ] + 1;
			}
		}
		for( int i = SZ( cola ) - 1 ; i >= 0 ; --i ) {
			int u = cola[ i ];
			tam[ u ] = 1;
			for( auto v : G[ u ] ) {
				if( pa[ u ] == v ) continue;
				tam[ u ] += tam[ v ];
			}
		}
		REP( i , n ) chain[ i ] = -1;
		REP( i , SZ( cola ) ) {
			int u = cola[ i ];
			if( chain[ u ] != -1 ) continue;
			chain[ u ] = u;
			int v = u;
			while( 1 ){ 
				int next = -1;
				for( auto v : G[ u ] ) if( pa[ v ] == u )
					if ( next == -1 || tam[ next ] < tam[ v ] ) next = v;
				if( next == -1 ) break;
				chain[ next ] = chain[ u ];
				u = next;
			}
			int len = h[ u ] - h[ v ] + 1;
			vector< tipo > a( len );
		
			REP( it , len ) {
				a[ it ] = paW[ u ];
				u = pa[ u ];
			}
			
			reverse( all( a ) );
			st[ v ] = Node( a );
		}
		REP( i , Q ){
			int op;
			sc( op );
			if( op == 1 ){
				int u , v;
				ll x;
				sc( u ) , sc( v );
				scanf( "%I64d" , &x );
				u -- , v --;
				tipo ans = querynodos( u , v ).val;
				printf( "%I64d\n" , x / ans );
			}else{
				int edg;
				ll w;
				sc( edg );
				scanf( "%I64d" , &w );
				edg --;
				int u = orig[ edg ] , v = dest[ edg ];
				if( h[ u ] >= h[ v ] ) swap( u , v );
				
				int c = chain[ v ] , len = st[ c ].len;
				st[ c ].update( h[ v ] - h[ c ] , w );
			}
		}
	}
}

/////Centroid Decomposition
//Codeforces Round #190 (Div. 1) C. Ciel the Commander
//IOI '11 - Pattaya, Thailand "Race"

#include<bits/stdc++.h>
using namespace std;

#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)v.size())

#define mp make_pair
#define fi first
#define se second

#define N 200000
#define INF (1<<29)

typedef vector< int > vi;
typedef long long ll;

ll target;
vi G[ N + 5 ] , C[ N + 5 ];
 
void add( int u , int v , int w ){
	G[ u ].pb( v );
	C[ u ].pb( w );
}
int DP[ N + 5 ] , MAXI[ N + 5 ];
vi order;
void dfs( int u , int p = -1 ){
	order.pb( u );
	DP[ u ] = 1 , MAXI[ u ] = 0;
	for( int i = 0 ; i < SZ( G[ u ] ) ; ++i ){
		int v = G[ u ][ i ];
		if( v == p ) continue;
		dfs( v , u );
		DP[ u ] += DP[ v ];
		MAXI[ u ] = max( MAXI[ u ] , DP[ v ] );
	}
}
void update( map< ll , int > &minAt , int depth , ll d ){
	if( !minAt.count( d ) ) minAt[ d ] = depth;
	else minAt[ d ] = min( minAt[ d ] , depth );
}
void fillMap( int u , int p , int depth , ll d , map< ll , int > &minAt ){
	update( minAt , depth , d );
	for( int i = 0 ; i < SZ( G[ u ] ) ; ++i ){
		int v = G[ u ][ i ];
		if( v == p ) continue;
		fillMap( v , u , depth + 1 , d + C[ u ][ i ] , minAt );
	}
}

int solve( int root ){
	order.clear();
	dfs( root );
	root = -1;
	int n = SZ( order );
	for( int i = 0 ; i < n ; ++i ){
		int u = order[ i ];
		MAXI[ u ] = max( MAXI[ u ] , n - DP[ u ] );
		if( 2 * MAXI[ u ] <= n ){
			root = u;
			break;
		}
	}
	assert( root != -1 );
	map< ll , int > minAt;
	minAt[ 0 ] = 0;
	int ans = INF;
	for( int i = 0 ; i < SZ( G[ root ] ) ; ++i ){
		int v = G[ root ][ i ];
		int pos = find( all( G[ v ] ) , root ) - G[ v ].begin();
		ll d = C[ v ][ pos ];
		G[ v ].erase( G[ v ].begin() + pos );
		C[ v ].erase( C[ v ].begin() + pos );
		
		map< ll , int > mapa;
		fillMap( v , -1 , 0 , 0 , mapa );
		
		ans = min( ans , solve( v ) );
		
		for( map< ll , int > :: iterator it = mapa.begin() ; it != mapa.end() ; ++it ){
			ll x = target - (it -> fi) - d;
			if( minAt.count( x ) ) ans = min( ans , minAt[ x ] + (it -> se) + 1 );
		}
		
		
		for( map< ll , int > :: iterator it = mapa.begin() ; it != mapa.end() ; ++it ){
			ll x = (it -> fi) + d;
			update( minAt , (it -> se) + 1 , x );
		}
	}
	return ans;
}
int main(){
	int n , d;
	while( scanf( "%d" , &n ) == 1 ){
		for( int i = 0 ; i < N ; ++i ) G[ i ].clear() , C[ i ].clear();
		scanf( "%d" , &d );
		target = d;
		for( int i = 0 ; i < n - 1 ; ++i ){
			int u , v , w;
			scanf( "%d%d%d" , &u , &v , &w );
			add( u , v , w );
			add( v , u , w );
		}
		int ans = solve( 0 );
		if( ans >= INF ) puts( "-1" );
		else printf( "%d\n" , ans );
	}
}
///
########################## BELLMAN FORD #################################
int n , m , sz;
int orig[ N + 5 ] , dest[ N + 5 ] , peso[ N + 5 ];
void add_edge( int u , int v , int w ){
	orig[ sz ] = u ;
	dest[ sz ] = v;
	peso[ sz ] = w;
	sz++;
}
vi D;
bool relax( int k ){
	int u = orig[ k ] , v = dest[ k ] , w = peso[ k ];
	if( D[ u ] + w < D[ v ] ){
		D[ v ] = D[ u ] + w;
		return 1;
	}
	return 0;
}
bool bellmanFord(){
	D = vi( n , INF );
	D[ 0 ] = 0;
	REP( it , n - 1 )
		REP( i , m )
			relax( i );
	REP( i , m ) if( relax( i ) ) return 0;
	return 1;
}
int main()
{
	int cases , u , v , w ;
	sc( cases );
	while( cases -- ){
		sc( n ) , sc( m );
		sz = 0;
		REP( i , m ){
			sc( u ) , sc( v ) , sc( w );
			add_edge( u , v , w );
		}
		puts( bellmanFord() ? "not possible" : "possible" );
	}
}
//3531 LA - Word Rings
// Maximum mean cycle
#define INF 100000000000LL
#define N (30 * 30)
#define M 1000000

int X[ M + 5 ] , Y[ M + 5 ] , W[ M + 5 ];
int orig[ M + 5 ] , dest[ M + 5 ] ;
ld peso[ M + 5 ];
ld D[ N + 5 ];
int maxi , n , m , S;

bool relax( int k ){
	int u = orig[ k ] , v = dest[ k ] ;
	ld w = peso[ k ];
	if( D[ u ] + w < D[ v ] ){
		D[ v ] = D[ u ] + w;
		return 1;
	}
	return 0;
}

bool check( ld target ){
	REP( i , n ) D[ i ] = INF;	
	REP( i , m ){
		int u = X[ i ] , v = Y[ i ];
		ld w = W[ i ];
		//orig[ i ] = u;
		//dest[ i ] = v;
		if( u == S ) peso[ i ] = 0;
		else peso[ i ] = target - w;
	}
	D[ S ] = 0.0;
	REP( it , n - 1 )
		REP( i , m )
			relax( i );
	REP( i , m ) if( relax( i ) ) return 1;
	return 0;
}
// 16 , 26 pasa
ld search(){
	ld lo = 0.0 , hi = maxi;
	if( !check( lo ) ) return -1;
	REP( it , 15 ){
		ld mi = ( hi + lo )/2;
		if( check( mi ) ) lo = mi;
		else hi = mi;
	}
	return lo;
}

char s[ 10000 + 5 ];
int main(){
	int len;
	while( sc( len ) == 1 ){
		if( !len ) break;
		n = 26 * 26 + 1;
		m = 0;
		maxi = INT_MIN;
		REP( it , len ){
			scanf( "%s" , s );
			int leng = strlen( s );
			int p1 = s[ 0 ] - 'a';
			int p2 = s[ 1 ] - 'a';
			int p3 = s[ leng - 2 ] - 'a';
			int p4 = s[ leng - 1 ] - 'a';
			int x = p1 * 26 + p2, y = p3 * 26 + p4 , w = leng;

			orig[ m ] = X[ m ] = x;
			dest[ m ] = Y[ m ] = y;
			peso[ m ] = W[ m ] = w;
			maxi = max( maxi , w );
			m ++;
		}
		S = n - 1;
		REP( i , S ) {
			orig[ m ] = X[ m ] = S;
			dest[ m ] = Y[ m ] = i;
			peso[ m ] = W[ m ] = 0;
			m ++;
		}
		ld ans = search();
		if( abs( ans - (-1) ) < 1e-6 ) puts( "No solution." );
		else printf( "%.2f\n" , (double)ans );
	}
}

############################################## min_cost_arborescence ##################################################
//// O( n * m )  aparentemente :P 
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second 

#define MAX_V 1000

#define test puts( "****************test****************" );

typedef long long ll;
typedef pair< int , int > pii;
typedef vector< int > vi;



typedef double edge_cost;
edge_cost INF = INT_MAX;

int V,root,prev[MAX_V];
bool adj[MAX_V][MAX_V];
edge_cost G[MAX_V][MAX_V],MCA;
bool visited[MAX_V],cycle[MAX_V];

void add_edge(int u, int v, edge_cost c){
    if(adj[u][v]) G[u][v] = min(G[u][v],c);
    else G[u][v] = c;
    adj[u][v] = true;
}

void dfs(int v){
    visited[v] = true;
    
    for(int i = 0;i<V;++i)
        if(!visited[i] && adj[v][i])
            dfs(i);
}

bool check(){
    memset(visited,false,sizeof(visited));
    dfs(root);
    
    for(int i = 0;i<V;++i)
        if(!visited[i])
            return false;
    
    return true;
}

int exist_cycle(){
    prev[root] = root;
    
    for(int i = 0;i<V;++i){
        if(!cycle[i] && i!=root){
            prev[i] = i; G[i][i] = INF;
            
            for(int j = 0;j<V;++j)
                if(!cycle[j] && adj[j][i] && G[j][i]<G[prev[i]][i])
                    prev[i] = j;
        }
    }
    
    for(int i = 0,j;i<V;++i){
        if(cycle[i]) continue;
        memset(visited,false,sizeof(visited));
        
        j = i;
        
        while(!visited[j]){
            visited[j] = true;
            j = prev[j];
        }
        
        if(j==root) continue;
        return j;
    }
    
    return -1;
}

void update(int v){
    MCA += G[prev[v]][v];
    
    for(int i = prev[v];i!=v;i = prev[i]){
        MCA += G[prev[i]][i];
        cycle[i] = true;
    }
    
    for(int i = 0;i<V;++i)
        if(!cycle[i] && adj[i][v])
            G[i][v] -= G[prev[v]][v];
    
    for(int j = prev[v];j!=v;j = prev[j]){
        for(int i = 0;i<V;++i){
            if(cycle[i]) continue;
            
            if(adj[i][j]){
                if(adj[i][v]) G[i][v] = min(G[i][v],G[i][j]-G[prev[j]][j]);
                else G[i][v] = G[i][j]-G[prev[j]][j];
                adj[i][v] = true;
            }
            
            if(adj[j][i]){
                if(adj[v][i]) G[v][i] = min(G[v][i],G[j][i]);
                else G[v][i] = G[j][i];
                adj[v][i] = true;
            }
        }
    }
}

bool min_cost_arborescence(int _root){
    root = _root;
    if(!check()) return false;
    
    memset(cycle,false,sizeof(cycle));
    MCA = 0;
    
    int v;
    
    while((v = exist_cycle())!=-1)
        update(v);
    
    for(int i = 0;i<V;++i)
        if(i!=root && !cycle[i])
            MCA += G[prev[i]][i];
    
    return true;
}
int X[ MAX_V + 5 ] , Y[ MAX_V + 5 ];

int main(){
	while( sc( V ) == 1 ){
		if( !V ) break;
		REP( i , V ) sc( X[ i ] ) , sc( Y[ i ] );
		int k , minY = 10000;
		REP( i , V ) if( Y[ i ] < minY ) minY = Y[ i ] , k = i;
		double ans = 1e100;
		
		clr( adj , 0 );
		REP( i , V ) REP( j , V ) if( Y[ i ] <= Y[ j ] ) add_edge( i , j , hypot( X[ i ] - X[ j ] , Y[ i ] - Y[ j ] ) );
		
		if( min_cost_arborescence( k ) ) ans = min( ans , MCA );
		
		printf( "%.2f\n" , (double)ans );
	}
}
########################## EULERIAN PATH #################################
//10129_UVA Eulerian path detection in directed
#define MAXE 1000000
#define MAXV 1000000

int next[ MAXE + 5 ] , to[ MAXE + 5 ] , last[ MAXV + 5 ] , E;
void addEdge( int u , int v ){
    next[ E ] = last[ u ] , to[ E ] = v , last[ u ] = E++;
}
int vis[ MAXV + 5 ] , used[ MAXV + 5 ];
int in[ MAXV + 5 ] , out[ MAXV + 5 ] , cant = 0;

void dfs( int u ){
    if( vis[ u ] ) return;
    vis[ u ] = true;
    cant++;
    for( int e = last[ u ] ; e != -1 ; e = next[ e ] ) 
		dfs( to[ e ] );
}
int nodes[ MAXV + 5 ];
int main(){
	int cases , n = 26;
	cin >> cases;
	REP( tc , cases ){
		int m ;
		cin >> m;
		
		E = 0;
		clr( last , -1 );
		clr( in , 0 ) , clr( out , 0 );
		clr( used , 0 );
		
		int top = 0 ;
		REP( i , m ){
			string s;
			cin >> s;
			int u = s[ 0 ] - 'a' , v = s[ SZ( s ) - 1 ] - 'a';
			addEdge( u , v );  
			
			if( !used[ u ] ) nodes[ top++ ] = u , used[ u ] = 1;
            if( !used[ v ] ) nodes[ top++ ] = v , used[ v ] = 1;
			     
	    	in[ v ] ++;        
	        out[ u ] ++;
	    }
	    int odd = 0 , root = -1;
        clr( vis , 0 );
        cant = 0;
        
        REP( i , top ){
            int u = nodes[ i ];
            if( abs( (int)(in[ u ] - out[ u ]) ) == 1 ) odd++;
            else if( in[ u ] != out[ u ] ) odd = n + 5; 
            
            if( in[ u ] - out[ u ] == -1 ) root = u;
            else if( root == -1 && in[ u ] == out[ u ] ) root = u;     
        }
        
        dfs( root );

        if( cant == top && odd <= 2 ) cout << "Ordering is possible." << '\n';
        else cout << "The door cannot be opened." << '\n';
	}
}
######################################################################################################################
//117_UVA Eulerian path detection in non-directed graph
#define N 26
#define INF (1<<29)

int ind[ N + 5 ] , dp[ N + 5 ][ N + 5 ];
int ans1 = 0;
void add( int u , int v , int w ){
	ind[ u ] ++ , ind[ v ] ++;
	dp[ u ][ v ] = dp[ v ][ u ] = w;
	ans1 += w;
}
int main(){
	string s;
	while( cin >> s ){
		if( s == "deadend" ) break;
		REP( i , N ) REP( j , N ) dp[ i ][ j ] = INF;
		REP( i , N ) dp[ i ][ i ] = 0;
		clr( ind , 0 );
		ans1 = 0;
		int n = SZ( s );
		add( s[ 0 ] - 'a' , s[ n - 1 ] - 'a' , n );
		while( cin >> s ){
			if( s == "deadend" ) break;
			int n = SZ( s );
			add( s[ 0 ] - 'a' , s[ n - 1 ] - 'a' , n );
		}
		int u = -1, v;
		REP( i , N ) if( ind[ i ] & 1 ){
			if( u == -1 ) u = i;
			else v = i;
		}
		REP( k , N ) REP( i , N ) REP( j , N ) dp[ i ][ j ] = min( dp[ i ][ j ] , dp[ i ][ k ] + dp[ k ][ j ] );
		if( u == -1 ) cout << ans1 << '\n';
		else cout << ans1 + dp[ u ][ v ] << endl;
	}
}
######################################################################################################################
//UVA 10054 Eulerian Path detection + reconstruction in non-directed


list< int > G[ N + 5 ]; // list for linear algorithm
list< int > L , LE;// save the solution , L ( nodes ) , LE( edges )
stack< int > S , SE; // stores the stacked nodes

int orig[ N + 5 ] , dest[ N + 5 ] , deg[ N + 5 ] , mapa[ N + 5 ] , id[ N + 5 ];
bool vis[ N + 5 ];
int n , m , ini , fin ;
bool connected(){
	queue< int >Q; 
	int cnt = 0;
	Q.push( 0 );///cuidado se esta asumiendo que la componente de aristas empieza en el nodo 0 puede suceder que existan nodos aislados
	clr( vis , 0 );
	vis[ 0 ] = 1;
	list< int > :: iterator it;
	while( !Q.empty() ){
		int u = Q.front();
		Q.pop();
		cnt++;
		for( it = G[ u ].begin() ; it != G[ u ].end() ; ++it ){
			int edge = *it;
			int v = orig[ edge ];
			if( u != dest[ edge ] ) v = dest[ edge ];
			if( !vis[ v ] ){
				vis[ v ] = 1 ; 
				Q.push( v );
			}
		}
	}	
	return cnt == n;
}
 
void delete_edge( int v , int w , int edge ){
	G[ v ].pop_front();
	list< int > :: iterator it;
	for( it = G[ w ].begin() ; it!= G[ w ].end() ; ++it ){
		if( *it == edge ){
			G[ w ].erase( it );
			break;
		}
	}
}
 
void ciclo( int v ){
	while( 1 ){
		if( G[ v ].empty() )break;
		int edge = G[ v ].front();
		int w = orig[ edge ];
		if( orig[ edge ] == v ) w = dest[ edge ];
		S.push( v ) , SE.push( edge ); 
		delete_edge( v , w  , edge ); 
		v = w;
	}
}
 
void solve( int start ){
	int v = start , edge;
	do{
		ciclo( v );
		v = S.top() , S.pop();
		edge = SE.top() , SE.pop();
		L.pb( v ) , LE.pb( edge ); 
	}while( !S.empty() );
}
 
void printCiclo(){
	list<int> :: iterator it1 , it2;
	vi v1( 1 , fin ) , v2;//v1 nodos , v2 aristas
	for( it1 = L.begin() ; it1 !=L.end() ; ++it1 )
		v1.pb( *it1 );
	for( it2 = LE.begin() ; it2 != LE.end() ; ++it2 )
		v2.pb( *it2 );
	
	REP( i , SZ( v2 ) ){
		bool ok;
		int edge = v2[ i ];
		printf( "%d %d\n" , id[ v1[ i ] ] , id[ v1[ i + 1 ] ] );
		//printf( "%d %d\n" , id[ v1[ i ] ] , id[ v1[ i + 1 ] ] );
		//printf("%d %c\n", edge + 1 ,  ok ? '+' : '-');
	}
}
 
int main(){
	int cases;
	sc( cases );
	REP( tc , cases ){
		if( tc ) puts( "" );
		sc( m );
		clr( mapa , -1 );
		clr( id , -1 );
		clr( deg , 0 );
		REP( i , N ) G[ i ].clear();
		L.clear() , LE.clear();
		
		n = 0;
		REP( i , m ){
			int x , y;
			sc( x ) , sc( y );
			if( mapa[ x ] == -1 ) mapa[ x ] = n , id[ n ++ ] = x;
			if( mapa[ y ] == -1 ) mapa[ y ] = n , id[ n ++ ] = y;
			x = mapa[ x ]  , y = mapa[ y ]; 
			deg[ x ]++ , deg[ y ]++;
			G[ x ].pb( i );
			G[ y ].pb( i );
			orig[ i ] = x , dest[ i ] = y;
		}
		int imp = 0 ;
		ini = -1 , fin =-1;
		REP( i , n ){
			if( deg[ i ] & 1 ){
				imp++;
				if( ini == -1 ) ini = i;
				else fin = i ;
			}
		}
		printf( "Case #%d\n" , tc + 1 );
		if( imp == 0 && connected() ){
			if( ini == -1 ) ini = 0 , fin = 0 ; //only cycle not eulerian path
			solve( ini );	
			printCiclo();	
		}else puts( "some beads may be lost" );
	}
}
######################################################################################################################
//Codeforces Round #288 (Div. 2)D. Tanya and Password
//Eulerian path reconstruction in directed graph O( E + V )  awesome!!

int next[ MAXE + 5 ] , to[ MAXE + 5 ] , last[ N + 5 ] , E;

void add( int u , int v ){
    next[ E ] = last[ u ] , to[ E ] = v , last[ u ] = E++;
}
bool vis_edge[ MAXE + 5 ];
int res[ MAXE + 5 ] , len;

void solve( int u ){
    for( int e = last[ u ] ; e != -1 ; e = next[ e ] ){
        int v = to[ e ];
        last[ u ] = next[ e ];
        if( vis_edge[ e ] ) break;
        vis_edge[ e ] = true;

        solve( v );
        res[ len++ ] = v;    
    }
} 

bool vis[ N + 5 ];
int in[ N + 5 ] , out[ N + 5 ] , cant;

void dfs( int u ){
    if( vis[ u ] ) return;
    vis[ u ] = 1;

    cant++;
    for( int e = last[ u ] ; e != -1; e = next[ e ] ) dfs( to[ e ] );
}
int used[ N + 5 ];
int main(){
	ios_base :: sync_with_stdio( 0 );
    int n;
    while( cin >> n ){
    	vi nodes ;
        clr( last , -1 );
        E = 0;
		clr( used , 0 );
        REP( i , n ){
        	string s;
            cin >> s;
            int u = s[ 0 ] * 300 + s[ 1 ];
            int v = s[ 1 ] * 300 + s[ 2 ];
            
            add( u , v );
            if( !used[ u ] ) nodes.pb( u ) , used[ u ] = 1;
            if( !used[ v ] ) nodes.pb( v ) , used[ v ] = 1;
            
            in[ v ]++;
            out[ u ]++;
        }
        
        int ip = 0 , ini = -1;
        
        REP( i , SZ( nodes ) ){
            int u = nodes[ i ];
            if( abs( in[ u ] - out[ u ] ) == 1 ) ip++;
            else if( in[ u ] != out[ u ] ) ip = 100; 
            
            if( in[ u ] - out[ u ] == -1 ) ini = u;
            else if( ini == -1 && in[ u ] == out[ u ] ) ini = u;     
        }
        
        cant = 0;
        clr( vis , 0 );
		if( ini != -1 ) dfs( ini );
        if( cant == SZ( nodes ) && ip <= 2 ){
            cout << "YES\n"; 
            len = 0;
            clr( vis_edge , 0 );
            solve( ini );
            
            cout << char( ini / 300 );
            cout << char( ini % 300 );
            for(int i = n - 1; i >= 0; i--) cout << char( res[ i ] % 300 );
            cout << '\n';
        }
        else cout << "NO\n";
    }
}
###################################################################
//UVA 10805 - Cockroach Escape Networks O(nm)
//SPOJ MDST
//Minimum distance spanning tree
#define N 25
#define INF (1<<29)

typedef pair< int , int > pii;
typedef vector< int > vi;
typedef vector< pii > vpii;

vi G[ N + 5 ] , G2[ N + 5 ];
int dist[ N + 5 ][ N + 5 ] , prev[ N + 5 ][ N + 5 ] , orig[ 100000 ] , dest[ 100000 ];

void add( vi g[ N + 5 ] , int u , int v ){
	g[ u ].pb( v );
	g[ v ].pb( u );
}

int bfs( int n , int s , int &dd ){
	queue< int >Q;
	vi d( n , INF );
	d[ s ] = 0;
	Q.push( s );
	int u;
	
	while( !Q.empty() ){
		u = Q.front();
		Q.pop();
		REP( i , SZ( G[ u ] ) ){
			int v = G[ u ][ i ] ;
			if( d[ v ] >= INF ){
				d[ v ] = d[ u ] + 1;
				Q.push( v );
			}
		}
	}
	dd = d[ u ];
	return u;
}

int getDiam( int n ){
	int diam;
	int a = bfs( n , 0 , diam );
	bfs( n , a , diam );
	return diam;
}
int main(){
	int cases;
	sc( cases );
	REP( tc , cases ){
		int n , m;
		sc( n ) , sc( m );
		REP( i , N ) G2[ i ].clear();
		REP( i , N ) REP( j , N ) dist[ i ][ j ] = INF;
		REP( i , m ){
			int u , v;
			sc( u ) , sc( v );
			add( G2 , u , v );
			orig[ i ] = u;
			dest[ i ] = v;
		}
		REP( s , n ){
			queue< int > Q;
			dist[ s ][ s ] = 0;
			Q.push( s );
			while( !Q.empty() ){
				int u = Q.front();
				Q.pop();
				REP( i , SZ( G2[ u ] ) ){
					int v = G2[ u ][ i ];
					if( dist[ s ][ v ] >= INF ){
						dist[ s ][ v ] = dist[ s ][ u ] + 1;
						prev[ s ][ v ] = u;
						Q.push( v );
					}
				}
			}
		}
		int ans = INF;
		REP( i , n ){
			REP( j , n ) G[ j ].clear() ;
			REP( j , n ) if( j != i ){
				add( G , prev[ i ][ j ] , j );
			}
			ans = min( ans , getDiam( n ) );
		}
		
		REP( i , m ){
			REP( j , n ) G[ j ].clear() ;
			int u = orig[ i ] , v = dest[ i ];
			add( G , u , v );
			REP( j , n ) {
				if( j == u ) continue;
				if( j == v ) continue;
				if( dist[ u ][ j ] < dist[ v ][ j ] ) add( G , prev[ u ][ j ] , j );
				else add( G , prev[ v ][ j ] , j );
			}
			ans = min( ans , getDiam( n ) );
		}
		
		printf( "Case #%d:\n" , tc + 1 );
		printf( "%d\n" , ans );
		puts( "" );
	}
}
###################################################################
################# FLOWS #######################################
###################################################################

########################## BIPARTITE MATCHING (sonnycson) #################################
//Short Code
// SPOJ 660. Dungeon of Death ( Vertex Cover in bipartite graph ) 
//http://en.wikipedia.org/wiki/K%C3%B6nig's_theorem_(graph_theory)
//maximum independent set + maxflow = nodes
//maximum independent set = minimun edge cover
//maxflow = minimum vertex cover
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define N 125

typedef vector< int > vi;
typedef long long ll;

int match[ N + 5 ];
bool vis[ N + 5 ];
vi G[ N + 5 ];
int dfs( int u ){
	if( vis[ u ] ) return 0;
	vis[ u ] = 1;
	for( auto v : G[ u ] ){
		if( match[ v ] == -1 || dfs( match[ v ] ) ){
			match[ v ] = u;
			return 1;
		}
	}
	return 0;
}
int main(){
	int cases , m , u , v ;
	sc( cases );
	REP( i , cases ){
		REP( i , N ) G[ i ].clear();
		sc( m );
		REP( i , m ){
			sc( u ) , sc( v );
			G[ u ].pb( v );
		}
		clr( match , -1 );
		int ans = 0 ;
		REP( i , N ){
			clr( vis , 0 );
			ans += dfs( i );
		}
		printf( "%d\n" , ans );
	}
}

// Insert and erase Edges
// SPOJ 11274. Lights and Switches
#define N 55
#define INF (1<<30)

int n,m;
int A[ N ][ N ] ;
pair< int , pii >V[ N*N ];
int match[ N ] , vis[ N ] , used[ N ];
vi M[N];
bool dfs( int u ){
	if( vis[u] )return 0;
	vis[u] = 1;
	FOR( v , M[u] )
		if( ( match[*v] == -1 || dfs( match[*v] ) ) )
		{
			match[*v] = u;
			used[u] = 1;
			return 1; 
		}
	return 0;
}
int bpm(){
	//clr( match , -1 );
	int cnt = 0;
	REP( k  , n )
		if( !used[k] )
		{
			clr( vis , 0 );
			cnt += dfs( k );
		}
		else cnt++;
	return cnt;
}
int main(){
	int tc;
	sc( tc );
	REP( t , tc ){
		sc( n );
		m = 0;
		REP( i ,  n )M[i].clear();
		REP( i , n )REP( j , n )sc( A[i][j] ) , V[m++] = mp( A[i][j] , pii( i , j ) );
		sort( V , V + m );
		int mini = INF;
		int lo = 0 , hi = 0;
		clr( match , -1 );
		clr( used , 0 );
		while( lo < m )
		{
			int BPM = bpm();
			while( hi < m && BPM < n )
			{
				M[ V[hi].se.fi ].pb( V[hi].se.se );//insert an edge
				hi++;
				BPM = bpm();
			}
			if( BPM == n )mini = min( mini , V[hi-1].fi - V[lo].fi );
			//erase an edge
			M[V[lo].se.fi].erase( M[V[lo].se.fi].begin() );
			used[ match[ V[lo].se.se ] ] = 0;
			match[ V[lo].se.se ] = -1;
			lo++;//
		}
		printf( "%d\n" , mini );
	}
}

##########################  BIPARTITE MATCHING (Fastest by Shinta) #################################
### BIPARTITE MATCHING - HOPCROFT-KARP O(E * sqrt(V))
//http://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm
// 12168 - Cat vs. Dog (UVA)
#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include<queue>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; i++ )

#define pb push_back

#define mp make_pair
#define fi first
#define se second

#define sync ios_base::sync_with_stdio(false);


typedef pair< int , int > pii;
typedef vector< int > vi;

int toi( string s ){ istringstream in( s ) ; int x ; in >> x; return x; }

#define N 500
#define M 500
#define EDG 250000

int le[ N + 5 ], ri[ M + 5 ] , distx[ N + 5 ] , disty[ M + 5 ];
int to[ EDG + 5 ], head[ N + 5 ] , next[ EDG + 5 ] , edg;
int n , m;

void reset(){
	edg = 0;
	REP( i , n ) head[ i ] = -1;
	REP( i , n ) le[ i ] = -1;
	REP( i , m ) ri[ i ] = -1;
}
void add(int u, int v){
    to[edg] = v; next[edg] = head[u]; head[u] = edg++;
}
bool bfs(){
    bool flag = false;
    REP(i, n) distx[i] = 0;
    REP(i, m) disty[i] = 0;
    
	queue< int >q;
    REP(i, n) if(le[i] == -1) q.push( i );
    while( !q.empty() ){
        int u = q.front(); q.pop();
        for(int e = head[u]; e != -1; e = next[e]){
            int v = to[e];
            if(!disty[v]){
                disty[v] = distx[u] + 1;
                if(ri[v] == -1) flag = true;
                else distx[ri[v]] = disty[v] + 1, q.push( ri[v] );
            }
        }
    }
    return flag;
}
bool dfs(int u){
    for(int e = head[u]; e != -1; e = next[e]){
        int v = to[e];
        if(disty[v] == distx[u] + 1){
            disty[v] = 0;
            if(ri[v] == -1 || dfs(ri[v])){
                le[u] = v;
                ri[v] = u;
                return true;
            }
        }
    }
    return false;
}

int match(){
    int res = 0;
    while(bfs())
        REP( i , n )
            if(le[i] == -1 && dfs(i)) res++;
    return res;
}


int main(){
	sync
	string s , t;
	int cases , cats , dogs , voters;
	cin >> cases;
	REP( tc , cases ){
		cin >> cats >> dogs >> voters;
		vector< pii > catLovers , dogLovers;
		REP( i , voters ){
			cin >> s >> t;
			int a = toi( s.substr( 1 ) ) , b = toi( t.substr( 1 ) );
			a-- , b--;
			if( s[ 0 ] == 'C' )
				catLovers.pb( mp( a , b ) );
			else
				dogLovers.pb( mp( a , b ) );
		}
		n = catLovers.size();
		m = dogLovers.size();
		reset();
		REP( i , n ) REP( j , m )
			if( catLovers[ i ].se == dogLovers[ j ].fi || catLovers[ i ].fi == dogLovers[ j ].se )
				add( i , j );
		int ans = n + m - match();
		cout << ans << '\n';
	}
}



########################## NON - BIPARTITE MATCHING (CHEN) #################################
//1099_TI
#define MAXN 222
int n;
bool adj[MAXN][MAXN];
int p[MAXN],m[MAXN],d[MAXN],c1[MAXN], c2[MAXN];
int q[MAXN], *qf, *qb;

int pp[MAXN];
int f(int x) {return x == pp[x] ? x : (pp[x] = f(pp[x]));}
void u(int x, int y) {pp[f(x)] = f(y);}

int v[MAXN];

void path(int r, int x){
    if (r == x) return;

    if (d[x] == 0){
        path(r, p[p[x]]);
        int i = p[x], j = p[p[x]];
        m[i] = j; m[j] = i;
    }
    else if (d[x] == 1){
        path(m[x], c1[x]);
        path(r, c2[x]);
        int i = c1[x], j = c2[x];
        m[i] = j; m[j] = i;
    }
}

int lca(int x, int y, int r){
    int i = f(x), j = f(y);
    while (i != j && v[i] != 2 && v[j] != 1){
        v[i] = 1; v[j] = 2;
        if (i != r) i = f(p[i]);
        if (j != r) j = f(p[j]);
    }
    
    int b = i, z = j;
    if(v[j] == 1) swap(b, z);

    for (i = b; i != z; i = f(p[i])) v[i] = -1;
    v[z] = -1;
    return b;
}

void shrink_one_side(int x, int y, int b){
    for(int i = f(x); i != b; i = f(p[i])){
        u(i, b);
        if(d[i] == 1) c1[i] = x, c2[i] = y, *qb++ = i;
    }
}

bool BFS(int r){
    for(int i=0; i<n; ++i)
      pp[i] = i;
    
    memset(v, -1, sizeof(v));
    memset(d, -1, sizeof(d));
    
    d[r] = 0;
    qf = qb = q;
    *qb++ = r;

    while(qf < qb){
        for(int x=*qf++, y=0; y<n; ++y){
            if(adj[x][y] && m[y] != y && f(x) != f(y)){
                if(d[y] == -1){
                    if(m[y] == -1){
            path(r, x);
            m[x] = y; m[y] = x;
            return true;
                    }
                    else{
            p[y] = x; p[m[y]] = y;
            d[y] = 1; d[m[y]] = 0;
            *qb++ = m[y];
                    }
                }
                else if(d[f(y)] == 0){
          int b = lca(x, y, r);
          shrink_one_side(x, y, b);
          shrink_one_side(y, x, b);
                }
          }
    }
  }
  
    return false;
}

int match(){
    memset(m, -1, sizeof(m));
    int c = 0;
    for (int i=0; i<n; ++i)
        if (m[i] == -1)
            if (BFS(i)) c++;
            else m[i] = i;
    
    return c;
}

int main()
{
	adj[ u ][ v ] = adj[ v ][ u ] = 1;
	cout << match() << endl;
}
#-----------------------------------------------------------------#
########################## MAX FLOW #################################
//http://en.wikipedia.org/wiki/Maximum_flow_problem
########################## FORD - FULKERSON #################################
//UVA 820 - Internet Bandwidth
#define INF (1<<30)
#define N 100
#define MAXV 201
#define MAXE 40803

int n , m , s , t , E;

int last[ MAXV + 5 ] , vis[ MAXV + 5 ];
int to[ 2*MAXE + 5 ] , cap[ 2*MAXE + 5 ] , next[ 2*MAXE + 5 ];
void add_edge( int u , int v , int uv , int vu = 0 ){
	to[ E ] = v ; cap[ E ] = uv ; next[ E ] = last[ u ]; last[ u ] = E++;
	to[ E ] = u ; cap[ E ] = vu ; next[ E ] = last[ v ]; last[ v ] = E++;
}

int dfs( int u , int f ){
	if( u == t ) return f;
	if( vis[ u ] ) return 0;
	
	vis[ u ] = 1;
	for( int e = last[ u ] ; e != -1 ; e = next[ e ] ){
		int v = to[ e ];
		if( cap[ e ] ){
			int ret = dfs( v , min( f , cap[ e ] ) );
			if( ret ){
				cap[ e ] -= ret;
				cap[ e ^ 1 ] += ret;
				return ret;
			}
		}
	}
	return 0;
}
int maxFlow(){
	int flow = 0;
	while( 1 ){
		REP( i , n ) vis[ i ] = 0;
		int f = dfs( s , INF );
		if( !f ) break;
		flow += f;
	}
	return flow;
}
#-----------------------------------------------------------------#      
//Codeforces Round #305 (Div. 1) D. Mike and Fish
//Circulation problem , maxflow with lowerbounds , upperbounds , demands       
########################## DINIC #################################
#define INF (1<<30)
#define N 200000

#define DEBUG( x ) cout << #x << " " << (x) << endl;

typedef vector< int > vi;
typedef long long ll;

struct flowGraph{
	int n , m , s , t , E;
	vi to , cap , NEXT;//maxe * 2
	vi last , now , dist;// maxv
	flowGraph(){}
	flowGraph( int n , int m , int s , int t ) : n( n ) , m( m ) , s( s ) , t( t ) {
		to = cap = NEXT = vi( 2 * m + 5 );
		last = now = dist = vi( n + 5 );
		E = 0;
		last = vi( n + 5 , -1 );
	}
	void add( int u , int v , int uv , int vu = 0 ){
		to[ E ] = v ; cap[ E ] = uv ; NEXT[ E ] = last[ u ] ; last[ u ] = E ++;
		to[ E ] = u ; cap[ E ] = vu ; NEXT[ E ] = last[ v ] ; last[ v ] = E ++;
	}
	bool bfs(){
		REP( i , n ) dist[ i ] = INF;
		queue< int > Q;
		dist[ t ] = 0;
		Q.push( t );
		while( !Q.empty() ){
			int u = Q.front() ; Q.pop();
			for( int e = last[ u ] ; e != -1 ; e = NEXT[ e ] ){
				int v = to[ e ];
				if( cap[ e ^ 1 ] && dist[ v ] >= INF ){
					dist[ v ] = dist[ u ] + 1;
					Q.push( v );
				}
			}
		}
		return dist[ s ] < INF;
	}
	int dfs( int u , int f ){
		if( u == t ) return f;
		for( int &e = now[ u ] ; e != -1 ; e = NEXT[ e ] ){
			int v = to[ e ];
			if( cap[ e ] && dist[ u ] == dist[ v ] + 1 ){
				int ret = dfs( v  , min( f , cap[ e ] ) );
				if( ret ){
					cap[ e ] -= ret;
					cap[ e ^ 1 ] += ret;
					return ret;
				}
			}
		}
		return 0;
	}
	ll maxFlow(){
		ll flow = 0;
		while( bfs() ){
		
			REP( i , n ) now[ i ] = last[ i ];
			while( 1 ){
				int f = dfs( s , INF );
				if( !f ) break;
				flow += f;
			}
		}
		return flow;
	}
};
void buildCirculationGraph( int n , vi &from , vi &to , vi &lo , vi &hi , int points ){
	int s = n , t = n + 1;
	flowGraph G( n + 2 , 1000000 , s , t );
	vi d( n + 2 );
	REP( i , SZ( from ) ){
		G.add( from[ i ] , to[ i ] , hi[ i ] - lo[ i ] );
		d[ from[ i ] ] += lo[ i ];
		d[ to[ i ] ] -= lo[ i ];
	}
	//For each v with d(v) < 0, add arc (s, v) with capacity -d(v). 
    //For each v with d(v) > 0, add arc (v, t) with capacity d(v).
	REP( i , n )
		if( d[ i ] < 0 ) G.add( s , i , -d[ i ] );
		else if( d[ i ] > 0 ) G.add( i , t , d[ i ] );
	int f = G.maxFlow();
	/*
	bool solve = 1;
    for( int e = last[ s ] ; e != -1 ; e = next[ e ] )
        if( cap[ e ] ) {
            solve = 0;
            break;
        }
    puts( solve ? "YES" : "NO" );
	*/
	REP( i , points ){
		if( (hi[ i ] - G.cap[ 2 * i ]) == 1 ) { // reconstructing flow
			putchar( 'r' );
		}else putchar( 'b' );
	}
	puts( "" );
}
vi UNIQUE( vi x ){
	sort( all( x ) );
	x.resize( unique( all( x ) ) - x.begin() );
	return x;
}
int getId( vi &v , int x ){
	return lower_bound( all( v ) , x ) - v.begin();
}
int main(){
	int n;
	while( sc( n ) == 1 ){
		vi X , Y;
		REP( i , n ){
			int x , y;
			sc( x ) , sc( y );
			X.pb( x ) , Y.pb( y );
		}
		vi A = UNIQUE( X ) , B = UNIQUE( Y );
		int nodes = SZ( A ) + SZ( B ) + 2 , s = nodes - 1 , t = s - 1;
		vi deg( nodes );
		vi from , to , lo , hi;
		REP( i , n ) {
			int u = getId( A , X[ i ] ) , v = SZ( A ) + getId( B , Y[ i ] );
			from.pb( u ) , to.pb( v ) , 
			lo.pb( 0 ) , hi.pb( 1 );
			deg[ u ] ++ , deg[ v ] ++;
		}
		REP( i , SZ( A ) ){
			int u = i;
			from.pb( s ) , to.pb( u ) , 
			lo.pb( deg[ u ]/2 ) , hi.pb( (deg[ u ] - 1)/2 + 1);
		}
		REP( i , SZ( B ) ){
			int v = SZ( A ) + i;
			from.pb( v ) , to.pb( t ) , 
			lo.pb( deg[ v ]/2 ) , hi.pb( (deg[ v ] - 1)/2 + 1);
		}
		from.pb( t ) , to.pb( s ) , lo.pb( 0 ) , hi.pb( 1000000 );
		buildCirculationGraph( nodes , from , to , lo , hi , n );
	}
}
//UVA 11419 - SAM I AM
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second
 
#define N 1000
#define MAXE 1100000
#define MAXV 2000

#define INF (1<<28)

#define test puts( "******************test**********************" );

typedef vector< int > vi;

int n , s , t , E = 0;
int to[ 2*MAXE + 5 ] , cap[ 2*MAXE + 5 ] , next[ 2*MAXE + 5 ] ; 
int dist[ MAXV + 5 ] , now[ MAXV + 5 ] , last[ MAXV + 5 ];
void addEdge( int u , int v , int uv , int vu = 0 ){
	to[ E ] = v ; cap[ E ] = uv ; next[ E ] = last[ u ]; last[ u ] = E ++;
	to[ E ] = u ; cap[ E ] = vu ; next[ E ] = last[ v ]; last[ v ] = E ++; 
}
bool bfs(){
	REP( i , n ) dist[ i ] = INF;
	queue< int > Q;
	dist[ t ] = 0;
	Q.push( t );
	while( !Q.empty() ){
		int u = Q.front() ; Q.pop();
		for( int e = last[ u ] ; e != -1 ; e = next[ e ] ){
			int v = to[ e ];
			if( cap[ e ^ 1 ] && dist[ v ] >= INF ){
				dist[ v ] = dist[ u ] + 1;
				Q.push( v );
			}
		}
	}
	return dist[ s ] < INF;
}
int dfs( int u , int f ){
	if( u == t ) return f;
	for( int &e = now[ u ] ; e != -1 ; e = next[ e ] ){
		int v = to[ e ];
		if( cap[ e ] && dist[ u ] == dist[ v ] + 1 ){
			int ret = dfs( v , min( f , cap[ e ] ) );
			if( ret ){
				cap[ e ] -= ret;
				cap[ e ^ 1 ] += ret;
				return ret;
			}
		}
	}
	return 0;
}
int maxFlow(){
	int flow = 0;
	while( bfs() ){
		REP( i , n ) now[ i ] = last[ i ];
		while( 1 ){
			int f = dfs( s , INF );
			if( !f ) break;
			flow += f;
		}
	}
	return flow;
}
int vis[ MAXV + 5 ];
void dfs( int u ){
	vis[ u ] = 1;
	for( int e = last[ u ] ; e != -1 ; e = next[ e ] ){
		int v = to[ e ];
		if( !vis[ v ] && cap[ e ] ) dfs( v );
	}
}
int main(){
	int R , C , m , r , c ;
	while( sc( R ) == 1 ){
		sc( C ) , sc( m );
		if( !R && !C && !m ) break;
		
		s = R + C , t = s + 1;
		n = t + 1;
		E = 0;
		REP( i , n ) last[ i ] = -1;
		
		REP( i , R ) addEdge( s , i , 1 );
		REP( i , C ) addEdge( R + i , t , 1 );
		
		REP( i , m ){
			sc( r ) , sc( c );
			r -- , c --;
			addEdge( r , R + c , INF );
		}
		printf( "%d" , maxFlow() );
		
		clr( vis , 0 );
		dfs( s );
		REP( i , R ) if( !vis[ i ] ) printf( " r%d" , i + 1 );
		REP( i , C ) if( vis[ R + i ] ) printf( " c%d" , i + 1 );
		puts( "" );
	}
}
#-----------------------------------------------------------------#   
//// how to find the cut 100200A_GYM
int vis[ MAXV + 5 ];

void dfs( int u ){
	vis[ u ] = 1;
	for( int e = last[ u ] ; e != -1 ; e = next[ e ] ){
		int v = to[ e ];
		if( !vis[ v ] && cap[ e ] ) dfs( v );
	}
}

vi findCut(){
	clr( type , 0 );
	
	clr( vis , 0 );
	dfs( s );

	vi vec;
	REP( e , 2 * m ) if( vis[ from[ e ] ] && !vis[ to[ e ] ] ) vec.pb( e );
	return vec;
}
////
########################## MAX FLOW MIN COST #################################
//Codeforces Round #212 (Div. 2) E. Petya and Pipes

typedef int Flow;
typedef int Cost;
const Flow INF = 0x3f3f3f3f;
struct Edge {
    int src, dst;
    Cost cst;
    Flow cap;
    int rev;
    Edge(){}
    Edge( int src , int dst , Cost cst , Flow cap , int rev ) : src( src ) , dst( dst ) , cst( cst ) , cap( cap ) , rev( rev ){}
};
bool operator<(const Edge a, const Edge b) {
    return a.cst>b.cst;
}

typedef vector<Edge> Edges;
typedef vector<Edges> Graph;

void add_edge( Graph&G , int u , int v , Flow c , Cost l ) {
    G[u].pb( Edge( u , v , l , c , G[v].size() ) );
    G[v].push_back( Edge( v , u , -l, 0 , (int)G[u].size() - 1 ) );
}
// returns the max_flow_mincost with cost <= K

pair< Flow, Cost > flow( Graph &G , int s , int t , int K = INF ) {
    int n = G.size();
    Flow flow = 0;
    Cost cost = 0;
    while( 1 ) {
        priority_queue< Edge > Q;
        vector< int > prev( n , -1 ), prev_num( n , -1 );
        vector< Cost > length( n , INF );
        Q.push( Edge( -1 , s , 0 , 0 , 0 ) );
        prev[ s ] = s;
        while( !Q.empty() ) {
            Edge e = Q.top(); Q.pop();
            int v = e.dst;
            for ( int i = 0 ; i < (int) G[v].size() ; i++ ) {
                if ( G[v][i].cap > 0 && length[ G[v][i].dst ] > e.cst + G[v][i].cst ) {
                    prev[ G[v][i].dst ] = v;
                    Q.push( Edge( v, G[v][i].dst , e.cst + G[v][i].cst , 0 , 0 ) );
                    prev_num[ G[v][i].dst ] = i;
                    length[ G[v][i].dst  ] = e.cst + G[v][i].cst;
                }
            }
        }
        if( prev[t] < 0 ) return make_pair( flow , cost );
        Flow mi = INF;
        Cost cst = 0;
        for( int v = t ; v != s ; v = prev[v] ) {
            mi = min( mi , G[prev[v]][prev_num[v]].cap );
            cst += G[prev[v]][prev_num[v]].cst;
        }
	if( cst > K ) return make_pair(flow, cost);
	if( cst != 0 ) mi = min(mi, K/cst);
	K -= cst*mi;
        cost+=cst*mi;

        for ( int v = t ; v != s ; v = prev[v] ) {
            Edge &e = G[prev[v]][prev_num[v]];
            e.cap -= mi;
            G[ e.dst ][ e.rev ].cap += mi;
        }
        flow+=mi;
    }
}
########################## MAX FLOW MIN COST #################################
// For no Integer Cost ( long double  ld )
//10746 UVA - Crime Wave - The Sequel
typedef int Flow;
typedef ld Cost;
const Flow INF = 0x3f3f3f3f;
struct Edge {
    int src, dst;
    Cost cst;
    Flow cap;
    int rev;
    Edge(){}
    Edge( int src , int dst , Cost cst , Flow cap , int rev ) : src( src ) , dst( dst ) , cst( cst ) , cap( cap ) , rev( rev ){}
};
bool operator<(const Edge a, const Edge b) {
    return a.cst>b.cst;
}

typedef vector<Edge> Edges;
typedef vector<Edges> Graph;

void add_edge( Graph&G , int u , int v , Flow c , Cost l ) {
    G[u].pb( Edge( u , v , l , c , G[v].size() ) );
    G[v].push_back( Edge( v , u , -l, 0 , (int)G[u].size() - 1 ) );
}
pair< Flow, Cost > flow( Graph &G , int s , int t ) {
    int n = G.size();
    Flow flow = 0;
    Cost cost = 0;
    while( 1 ) {
        priority_queue< Edge > Q;
        vector< int > prev( n , -1 ), prev_num( n , -1 );
        vector< Cost > length( n , INF );
        Q.push( Edge( -1 , s , 0 , 0 , 0 ) );
        prev[ s ] = s;
        while( !Q.empty() ) {
            Edge e = Q.top(); Q.pop();
            int v = e.dst;
            for ( int i = 0 ; i < (int) G[v].size() ; i++ ) {
                if ( G[v][i].cap > 0 && length[ G[v][i].dst ] > e.cst + G[v][i].cst ) {
                    prev[ G[v][i].dst ] = v;
                    Q.push( Edge( v, G[v][i].dst , e.cst + G[v][i].cst , 0 , 0 ) );
                    prev_num[ G[v][i].dst ] = i;
                    length[ G[v][i].dst  ] = e.cst + G[v][i].cst;
                }
            }
        }
        if( prev[t] < 0 ) return make_pair( flow , cost );
        Flow mi = INF;
        Cost cst = 0;
        for( int v = t ; v != s ; v = prev[v] ) {
            mi = min( mi , G[prev[v]][prev_num[v]].cap );
            cst += G[prev[v]][prev_num[v]].cst;
        }


	    cost+=cst*mi;

        for ( int v = t ; v != s ; v = prev[v] ) {
            Edge &e = G[prev[v]][prev_num[v]];
            e.cap -= mi;
            G[ e.dst ][ e.rev ].cap += mi;
        }
        flow+=mi;
    }
}
########################## HUNGARIAN ALGORITHM n^3 #################################
https://www.topcoder.com/community/data-science/data-science-tutorials/assignment-problem-and-hungarian-algorithm/
//works also for maximum-weighted and minimum-weighted  change cost(x,y) por -cost(x,y)
//its calculing maximum
#define MAX_V 500

int V,cost[MAX_V][MAX_V];
int lx[MAX_V],ly[MAX_V];
int max_match,xy[MAX_V],yx[MAX_V],PREV[MAX_V];
bool S[MAX_V],T[MAX_V];
int slack[MAX_V],slackx[MAX_V];
int q[MAX_V],head,tail;

void init_labels(){
    memset(lx,0,sizeof(lx));
    memset(ly,0,sizeof(ly));
	
    for(int x = 0;x<V;++x)
        for(int y = 0;y<V;++y)
            lx[x] = max(lx[x],cost[x][y]);
}

void update_labels(){
    int x,y,delta = INT_MAX;
	
    for(y = 0;y<V;++y) if(!T[y]) delta = min(delta,slack[y]);
    for(x = 0;x<V;++x) if(S[x]) lx[x] -= delta;
    for(y = 0;y<V;++y) if(T[y]) ly[y] += delta;
    for(y = 0;y<V;++y) if(!T[y]) slack[y] -= delta;
}

void add_to_tree(int x, int prevx){
    S[x] = true;
    PREV[x] = prevx;
	
    for(int y = 0;y<V;++y){
        if(lx[x]+ly[y]-cost[x][y]<slack[y]){
            slack[y] = lx[x]+ly[y]-cost[x][y];
            slackx[y] = x;
        }
    }
}

void augment(){
    int x,y,root;
    head = tail = 0;
    memset(S,false,sizeof(S));
    memset(T,false,sizeof(T));
    memset(PREV,-1,sizeof(PREV));
    
    for(x = 0;x<V;++x){
        if(xy[x]==-1){
            q[tail++] = root = x;
            PREV[root] = -2;
            S[root] = true;
            break;
        }
    }
	
    for(y = 0;y<V;++y){
        slack[y] = lx[root]+ly[y]-cost[root][y];
        slackx[y] = root;
    }
	
    while(true){
        while(head<tail){
            x = q[head++];
			
            for(y = 0;y<V;++y){
                if(cost[x][y]==lx[x]+ly[y] && !T[y]){
                    if(yx[y]==-1) break;
					
                    T[y] = true;
                    q[tail++] = yx[y];
                    add_to_tree(yx[y],x);
                }
            }
			
            if(y<V) break;
        }
		
        if(y<V) break;
		
        update_labels();
        head = tail = 0;
		
        for(y = 0;y<V;++y){
            if(!T[y] && slack[y]==0){
                if(yx[y]==-1){
                    x = slackx[y];
                    break;
                }
				
                T[y] = true;
				
                if(!S[yx[y]]){
                    q[tail++] = yx[y];
                    add_to_tree(yx[y],slackx[y]);
                }
            }
        }
		
        if(y<V) break;
    }
	
    ++max_match;
	
    for(int cx = x,cy = y,ty;cx!=-2;cx = PREV[cx],cy = ty){
        ty = xy[cx];
        yx[cy] = cx;
        xy[cx] = cy;
    }
}

int hungarian(){
    int ret = 0;
    max_match = 0;
    memset(xy,-1,sizeof(xy));
    memset(yx,-1,sizeof(yx));
	
    init_labels();
    for(int i = 0;i<V;++i) augment();
    for(int x = 0;x<V;++x) ret += cost[x][xy[x]];
	
    return ret;
}
int main(){
	while( sc( V ) == 1 ){
		REP( i , V ) REP( j , V ) sc( cost[ i ][ j ] );
		printf( "%d\n" , hungarian() );
	}
}

##########################  ALL PAIRS MAXMIMUM FLOW #################################
// gomory hu tree
//11594_UVA
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define FOR(i,c) for(int i=0;i!=(c).size();++i)
#define RESIDUE(s,t) (capacity[s][t]-flow[s][t])

#define INF (1<<29)
#define MAXN 200

typedef int Weight;
typedef vector< int > vi;

// !!!!!!!! following must be initialized by user !!!!!!!!!!!
vi graph[ MAXN ]; 
Weight capacity[ MAXN ][ MAXN ];
int n; // # of vertices
// !!!!!!!!
vi ghtree[ MAXN ]; 
vector<Weight> ghweight[ MAXN ];
int p[ MAXN ] , prev[ MAXN ]; 
Weight w[ MAXN ] , flow[ MAXN ][ MAXN ];
Weight max_memo[ MAXN ][ MAXN ]; //dp table, initialize to -1

void build_tree() {
	REP( i , MAXN ) w[ i ] = 0;
	clr( p , 0 );
	for( int s = 1 ; s < n ; ++s ){
    	int t = p[ s ];
    	clr( flow , 0 );
    	Weight total = 0;
		while( 1 ){
			queue< int > q; 
			q.push( s );
			clr( prev , -1 );
			prev[ s ] = s;
      		while( !q.empty() && prev[ t ] < 0 ) {
				int u = q.front(); q.pop();
        		FOR( e , graph[ u ] ) if( prev[ graph[ u ][ e ] ] < 0 && RESIDUE( u , graph[ u ][ e ] ) > 0 )
					prev[ graph[ u ][ e ] ] = u, q.push( graph[ u ][ e ] );
      		}
      		if( prev[ t ] < 0 )break;
      		Weight inc = INF;
      		for( int j = t ; prev[ j ] != j ; j = prev[ j ] ) inc = min( inc , RESIDUE( prev[ j ] , j ) );
      		for( int j = t ; prev[ j ] != j ; j = prev[ j ] )
				flow[ prev[ j ] ][ j ] += inc , flow[ j ][ prev[ j ] ] -= inc;
      		total += inc;
		}
    	w[ s ] = total;
   		REP( u , n ) if( u != s && prev[ u ] != -1 && p[ u ] == t ) p[ u ] = s;
    	if( prev[ p[ t ] ] != -1 ) p[ s ] = p[ t ] , p[ t ] = s, w[ s ] = w[ t ] , w[ t ] = total; 
	}
	REP( i , MAXN ) ghtree[ i ].clear() , ghweight[ i ].clear();
	REP( s , n ) if( s != p[ s ] ) {
		ghtree[ s ].push_back( p[ s ] );
		ghtree[ p[ s ] ].push_back( s );
		ghweight[ s ].push_back( w[ s ] );
		ghweight[ p[ s ] ].push_back( w[ s ] ); 
	}
  	clr( max_memo , -1 ); 
}
Weight max_flow( int u , int t , int p = -1 ) {
	if( max_memo[ u ][ t ] != -1 ) return max_memo[ u ][ t ]; 
	if( u == t ) return INF - 1;
	Weight d = INF; 
	FOR( e , ghtree[ u ] ) 
		if( ghtree[ u ][ e ] != p ) {
    		Weight ans = max_flow( ghtree[ u ][ e ] , t , u );
			if( ans < INF ) ans = min( ans , ghweight[ u ][ e ] );
			d = min( d , ans );
		}
	if( d < INF ) max_memo[ u ][ t ] = max_memo[ t ][ u ] = d;
	return d; 
}
  
int main(){
	int cases;
	sc( cases );
	REP( tc , cases ){
		sc( n );
		REP( i , n ) graph[ i ].clear();
		REP( u , n ) REP( v , n ){
			int w;
			sc( w );
			graph[ u ].pb( v );
			capacity[ u ][ v ] = w;
		}
		build_tree();
		printf( "Case #%d:\n" , tc + 1 );
		REP( u , n ) REP( v , n ) printf( "%d%c" , u == v ? 0 : max_flow( u , v ) , v + 1 == n ? 10 : 32 );
	}
}

###################################################################
################# STRINGS #######################################
###################################################################

############################# Kmp ##############################
//SPOJ 263. Period
// T[ 0 ] is the empty string then T[ n ] is the board of string of size n

//  Matches may overlap
///// KMP SHORT by chen
// string s is 0 - based
vi f( string p , string t ){
	int np = p.size() , nt = t.size();
	vi T( np + 1 );
	int j = T[ 0 ] = -1;
	REP( i , np ){
		while( j != -1 && p[ i ] != p[ j ] ) j = T[ j ];
		T[ i + 1 ] = ++j;
	}
	//KMP matcher
	//SPOJ 32. A Needle in the Haystack
	
	vi v;
	j = 0;
	REP( i , nt ){
		while( j != -1 && ( j == np || t[ i ] != p[ j ] ) ) j = T[ j ];
		j ++;
		if( j == np ) v.pb( i - np + 1 );
	}
	return v;
}
//----------------------------
// Explanation of Period - Find the period of substring of len i
	if( i % (i - T[ i ]) == 0 && T[ i ] >= i/2 )
// consider this cases "a" , "aba" , "aa" , "aaa" , "ababa"


############################# Trie ###############################################
//http://www.codechef.com/MAY12/problems/TWSTR/ trie			??       
//Example : in a dp problem (2756 TJU)
#define N 20005
#define ALP 26

int next[ N ][ ALP ] , node ,term[ N ];

void init(){
	node = 1;
	clr( term , 0 );
	clr( next , 0 );
}
void add( char *s , int n ){
	int p = 0;
	REP( i , n ){
		if( !next[ p ][ s[ i ] ] ) next[ p ][ s[ i ] ] = node++;
		p = next[ p ][ s[ i ] ];
	}
	term[ p ] = 1;
}
int n;
char s[ N ] , t[ N ];
int memo[ 605 ][ N ];

int dp( int pos , int nodo ){
	if( pos == n ) return term[ nodo ]?0:INF;
	int &dev = memo[ pos ][ nodo ];
	if( dev == -1 ){
		dev = 1 + dp( pos + 1 , nodo );
		int sig = next[ nodo ][ s[ pos ] ];
		if( sig ){
			dev = min( dev , dp( pos + 1 , sig ) );//si existe siguiente en el trie escojo tomarlo
			if( term[ sig ] )//y si es termina puedo escoger desde el inicio de alguna palabra
				dev = min( dev , dp( pos + 1 , 0 ) );
		}// no hay else para no dirigirse a estados no validos
	}
	return dev;
}
#-----------------------------------------------------------------#
########################## Aho - Corasick #################################
// convert the array next to go of aho - corasick algorithm
//1158_TI_1
typedef unsigned char tint;

int next[ MAXNODES + 5 ][ ALP + 1 ],  term[ MAXNODES + 5 ] , T[ MAXNODES + 5 ];
int nAlp , n , nodes;
string alp;
void init(){
	nodes = 1;
	clr( next , 0 );
	clr( term , 0 );
}
void add( string &s ){
	int ns = s.size() , p = 0;
	REP( i , ns ){
		tint c = s[ i ];
		if( next[ p ][ c ] ) p = next[ p ][ c ];
		else p = next[ p ][ c ] = nodes ++;
	}
	term[ p ] = 1;
}
void aho(){
	queue< int > Q;
	REP( i , nAlp ){
		tint c = alp[ i ];
		int u = next[ 0 ][ c ];
		if( u ) T[ u ] = 0 , Q.push( u );
	}
	while( !Q.empty() ){
		int u = Q.front() ; Q.pop();
		REP( i , nAlp ){
			tint c = alp[ i ];
			int v = next[ u ][ c ] , x = next[ T[ u ] ][ c ];
			if( v == 0 ) next[ u ][ c ] = x;
			else Q.push( v ) , T[ v ] = x , term[ v ] |= term[ T[ v ] ];
		}
	}
}

// with adyancency list
//SUB_PROB SPOJ
vi V[ ND ];
vector< pair< char , int > > trie[ND];
int T[ ND ] , Node ;

inline int getNode( int node , char c )
{
	FOR( o , trie[node] )
		if( o->fi == c )return o->se;
	return 0;	
}
void add( char *s , int id )
{
	int ns = strlen( s ) , p = 0 ;
	REP( i , ns )
	{
		int v = getNode( p , s[i] );
		if( !v )
		{
			trie[p].pb( mp( s[i] , Node ) );
			p = Node++;
		}
		else p = v;
	}
	V[ p ].pb( id );
}

void aho()
{
	queue< int >Q;
	FOR( o , trie[0] )
		Q.push( o->se ) , T[ o->se ] = 0;
	while( !Q.empty() )
	{
		int u = Q.front();
		Q.pop();
		FOR( o , trie[u] )
		{
			int v = o->se;
			char c = o->fi;
			int p = T[u];
			while( p && getNode( p , c ) == 0 )p = T[p];
			p = getNode( p , c );
			T[ v ] = p;
			Q.push( v );
			FOR( q , V[ T[v] ] )V[ v ].pb( *q );
		}
	}
}

int ans[ M ];

int main()
{
	char s[ N ] , t[ M ];
	int n;
	scanf( "%s%d" , s , &n );
	Node = 1;
	REP( i , n ) scanf( "%s" , t ) , add( t , i );
	int ns = strlen( s );
	aho();
	int p = 0;
	REP( i , ns )
	{
		char c = s[i];
		while( p && getNode( p , c ) == 0 ) p = T[p];
		p = getNode( p , c );

		FOR( o , V[p] )ans[*o] = 1;
	}
	REP( i , n )puts( (ans[i]?"Y":"N") );
}

////CDF #311 (Div. 2) E
/// Find K-th string
#define N 5000
#define NODES (N * N)

typedef long long ll;
typedef vector< ll > vll;
typedef pair< int , int > pii;
typedef vector< int > vi;

bool used[ N + 5 ][ N + 5 ];
bool memo[ N + 5 ][ N + 5 ];
bool dp( int lo , int hi , string &s ){
	if( lo >= hi ) return 1;
	if( used[ lo ][ hi ] ) return memo[ lo ][ hi ];
	used[ lo ][ hi ] = 1;
	bool &dev = memo[ lo ][ hi ] = 0;
	if( s[ lo ] == s[ hi ] ) dev |= dp( lo + 2 , hi - 2 , s );
	return dev;
}
int node;
int F[ NODES + 5 ];
int NEXT[ 2 ][ NODES + 5 ];
void clear(){
	node = 1;
	clr( F , 0 );
	clr( NEXT , 0 );
}
int AC[ NODES + 5 ];
int dfs( int u ){
	AC[ u ] = F[ u ];
	REP( i , 2 ){
		int v = NEXT[ i ][ u ];
		if( v ){
			dfs( v );
			AC[ u ] += AC[ v ];
		}
	}
}
int main(){
	ios_base :: sync_with_stdio( 0 );
	string s;
	int K;
	while( cin >> s >> K ){
		clear();
		clr( used , 0 );
		int n = SZ(s);
		for( int i = 0 ; i < n ; ++i ){
			int p = 0;
			for( int j = i ; j < n ; ++j ){
				int cur = s[ j ] - 'a';
				if( !NEXT[ cur ][ p ] ) NEXT[ cur ][ p ] = node ++;
				p = NEXT[ cur ][ p ];
				if( dp( i , j , s ) ) F[ p ]++;
			}
		}
		dfs( 0 );
		int u = 0;
		s.clear();
		while( 1 ){
			if( K <= F[ u ] )
				break;
			int v = NEXT[ 0 ][ u ];
			int w = NEXT[ 1 ][ u ];
			if( v && K <= AC[ v ] + F[ u ] ){
				K -= F[ u ];
				u = v;
				s.pb( 'a' );
				continue;
			}
			K -= F[ u ];
			if( v ) K -= AC[ v ];
			if( w ){
				u = w;
				s.pb( 'b' );
			}else break;
		}
		cout << s << '\n';
	}
}
########################## HASHING #################################

ll Mod[] = { 1000000007 , 1000000009 };
ll Inv[] = { 370370373 , 296296299 } ; 
void add_R( int c )
{
	H = H*B + c ;
	HR = HR + Pot*c ; 
	Pot = Pot*B ;
}
void del_L( int c )
{
	Pot = Pot*Inv ;		
	H = H - c*Pot ;
	HR = ( HR - c )Inv ;
}
[X.....]Y
H = HB - XB^L + Y
HR = [ HR - X ]/B + YB^(L-1)

/////
void get( int i , int j , ll &h , ll &hr ){
	h = H[ j + 1 ] - H[ i ] * POT[ j - i + 1 ];
	fix( h , MOD );

	hr = ( HR[ j + 1 ] - HR[ i ] ) * INV[ i ];
	fix( hr , MOD );

}
	B1 = modInv( B , MOD ) ;
	ll h = 0 , hr = 0 , pot = 1;
	ll inv = 1;
	H[ 0 ] = HR[ 0 ] = 0; 
	POT[ 0 ] = INV[ 0 ] = 1 ;
	REP( i , n ){
		h = ( h * B + s[ i ] )%MOD;
		hr = ( hr + pot * (ll) s[ i ] )%MOD;
		pot = ( pot * B ) %MOD;
		inv = ( inv * B1 ) %MOD;
		H1[ i + 1 ] = h1;
		HR1[ i + 1 ] = hr1;
		POT1[ i + 1 ] = pot1;
		INV1[ i + 1 ] = inv1;
	}
/////////////////////

// hashing con overflow , usar ll o ull y dejar de usar modulos

######################## MANACHER ALGORITHM ###################################
http://e-maxx.ru/algo/palindromes_count
f returns # of diferents subpalindromes
d1[ i ] return # of palidromes of odd size with center i
d2[ i ] return # of palidromes of even size with center i , i - 1 
// StringHash by Egor     http://codeforces.com/contest/113/submission/676303/
ll pow( ll a , ll b , ll c ){
	ll ans = 1;
	while( b ){
		if( b&1 ) ans = ( ans * a )%c;
		a = ( a * a )%c;
		b >>= 1;
	}
	return ans;
}
ll modInverse( ll a , ll mod ){ return pow( a , mod - 2 , mod );}
struct StringHash{
	ll mod;
	vll hash , reversePower;
	StringHash(){}
	StringHash( string s , ll module )
	{
		ll B = 27;
		mod = module;
		ll RB = modInverse( B , mod );
		int n = s.size();
		hash.resize( n + 1 ) , reversePower.resize( n + 1 );
		reversePower[ 0 ] = 1;
        hash[ 0 ] = 0;
		ll power = 1;
		REP( i , n ){
			hash[ i + 1 ] = ( hash[ i ] + ( ( s[ i ] - 'a' + 1 ) * power )%mod ) %mod;
			power = ( power * B )%mod;
			reversePower[ i + 1 ] = ( reversePower[ i ] * RB )%mod;
		}
	}
	ll Hash( int from , int to ){
		return ( ( ( hash[ to ] - hash[ from ] + mod )% mod ) * reversePower[ from ] ) %mod;
	}
};

int cnt_unique_palin( string &s )
{
	int n = s.length();
	StringHash H1 = StringHash( s , 1000000009LL );
	StringHash H2 = StringHash( s , 1000000007LL );
	set< pll >S;
	vi d1( n );
	int l = 0 , r = -1;
	for( int i = 0 ; i < n ; ++i ) {
	    int k = ( i > r ? 1 : min ( d1[ l + r - i ] , r - i + 1 ) );
	    if( i > r )
	        S.insert( pll( H1.Hash( i , i + 1 )  , H2.Hash( i , i + 1 ) ) );
	    while( i + k < n && i - k >= 0 && s[ i - k ] == s[ i + k ] )  {
	        S.insert( pll( H1.Hash( i - k , i + k + 1 )  , H2.Hash( i - k , i + k + 1 ) ) );
	        k++;
	    }
	    d1[ i ] = k--;
	    if( i + k > r )
	            l = i - k , r = i + k ;
	}
	vi d2( n );
	l = 0 ; r = -1;
	for( int i = 0 ; i < n ; ++i ) {
	    int k = ( i > r ? 0 : min ( d2[ l + r - i + 1 ] , r - i + 1 ) );
	    while ( i + k < n && i - k - 1 >= 0 && s[ i - k - 1 ] == s[ i + k ] )  {
	        S.insert( pll( H1.Hash( i - k - 1 , i + k + 1 )  , H2.Hash( i - k - 1 , i + k + 1 ) ) );
	        k++;
	    }
	    d2[ i ] = k--;
	    if( i + k > r )
	            l = i - k - 1 ,  r = i + k;
	}
	return S.size();
}

######################## SUFFIX ARRAY ###################################
//LCP ( i , i + 1 )
// Cuidaddo con el lcp[ N - 1 ] , puede ser cualquier cosa
O( n logn^2 )
//http://www.spoj.pl/problems/SARRAY/
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )

#define N 200000

string s;
int n , gap;
int pos[ N + 5 ] , tmp[ N + 5 ] , sa[ N + 5 ] , lcp[ N + 5 ];

bool sufCmp( int i , int j ){
	if( pos[ i ] != pos[ j ] ) return pos[ i ] < pos[ j ];
	i += gap , j += gap;
	return (i < n && j < n) ? pos[ i ] < pos[ j ] : i > j ;
}
void build(){
	n = s.size();
	REP( i , n ) sa[ i ] = i , pos[ i ] = s[ i ];
	for( gap = 1 ; ; gap <<= 1 ){
		sort( sa , sa + n , sufCmp );
		REP( i , n - 1 ) tmp[ i + 1 ] = tmp[ i ] + sufCmp( sa[ i ] , sa[ i + 1 ] );
		REP( i , n ) pos[ sa[ i ] ] = tmp[ i ];
		if( tmp[ n - 1 ] == n - 1 ) break;
	}
	for( int i = 0 , k = 0 ; i < n ; ++i )
		if( pos[ i ] != n - 1 ){
			for( int j = sa[ pos[ i ] + 1 ] ; s[ i + k ] == s[ j + k ] ; ) ++k;
			lcp[ pos[ i ] ] = k;
			if( k ) --k;
		}
}
int rmq[ LOGN + 1 ][ N + 5 ];
void RMQ(){
	REP( i , n ) rmq[ 0 ][ i ] = lcp[ i ];
	for( int k = 1 ; k <= LOGN ; ++k )
		for( int i = 0 ; i + (1<<k) <= n ; ++i )
			rmq[ k ][ i ] = min( rmq[ k - 1 ][ i ] , rmq[ k - 1 ][ i + (1<<(k-1)) ] );
}
int query( int i , int j ){
	int r = 31 - __builtin_clz( j - i + 1 );
	return min( rmq[ r ][ i ] , rmq[ r ][ j - ( 1 << r ) + 1 ] );
}
int LCP( int i , int j ){
	return query( i , j - 1 );
}
//0 , i - 1
int search1( int lo , int hi , int target ){
	if( hi == -1 ) return 0;
	if( LCP( hi , hi + 1 ) < target ) return hi + 1;
	if( LCP( lo , hi + 1 ) == target ) return lo;
	while( hi - lo > 1 ){
		int med = ( lo + hi )>>1;
		if( LCP( med , hi + 1 ) == target ) hi = med;
		else lo = med;
	}
	return hi;
}
//i + 1 , n - 1
int search2( int lo , int hi , int target ){
	if( lo == n ) return n - 1;
	if( LCP( lo - 1 , lo ) < target ) return lo - 1;
	if( LCP( lo - 1 , hi ) == target ) return hi;
	while( hi - lo > 1 ){
		int med = (lo + hi) >>1;
		if( LCP( lo - 1 , med ) == target ) lo = med;
		else hi = med;
	}
	return lo;
}
##########################  Suffix Arrays O(n)#################################
// Begins Suffix Arrays implementation
// O(n log n) - Manber and Myers algorithm
// Refer to "Suffix arrays: A new method for on-line string searches",
// by Udi Manber and Gene Myers
 
//Usage:
// Fill str with the characters of the string.
// Call SuffixSort(n), where n is the length of the string stored in str.
// That's it!
 
//Output:
// pos = The suffix array. Contains the n suffixes of str sorted in lexicographical order.
//       Each suffix is represented as a single integer (the position of str where it starts).
// rank = The inverse of the suffix array. rank[i] = the index of the suffix str[i..n)
//        in the pos array. (In other words, pos[i] = k <==> rank[k] = i)
//        With this array, you can compare two suffixes in O(1): Suffix str[i..n) is smaller
//        than str[j..n) if and only if rank[i] < rank[j]
 
int str[N]; //input
int rank[N], pos[N]; //output
int cnt[N], next[N]; //internal
bool bh[N], b2h[N];
 
// Compares two suffixes according to their first characters
bool smaller_first_char(int a, int b){
  return str[a] < str[b];
}
 
void suffixSort(int n){
  //sort suffixes according to their first characters
  for (int i=0; i<n; ++i){
    pos[i] = i;
  }
  sort(pos, pos + n, smaller_first_char);
  //{pos contains the list of suffixes sorted by their first character}
 
  for (int i=0; i<n; ++i){
    bh[i] = i == 0 || str[pos[i]] != str[pos[i-1]];
    b2h[i] = false;
  }
 
  for (int h = 1; h < n; h <<= 1){
    //{bh[i] == false if the first h characters of pos[i-1] == the first h characters of pos[i]}
    int buckets = 0;
    for (int i=0, j; i < n; i = j){
      j = i + 1;
      while (j < n && !bh[j]) j++;
      next[i] = j;
      buckets++;
    }
    if (buckets == n) break; // We are done! Lucky bastards!
    //{suffixes are separted in buckets containing strings starting with the same h characters}
 
    for (int i = 0; i < n; i = next[i]){
      cnt[i] = 0;
      for (int j = i; j < next[i]; ++j){
        rank[pos[j]] = i;
      }
    }
 
    cnt[rank[n - h]]++;
    b2h[rank[n - h]] = true;
    for (int i = 0; i < n; i = next[i]){
      for (int j = i; j < next[i]; ++j){
        int s = pos[j] - h;
        if (s >= 0){
          int head = rank[s];
          rank[s] = head + cnt[head]++;
          b2h[rank[s]] = true;
        }
      }
      for (int j = i; j < next[i]; ++j){
        int s = pos[j] - h;
        if (s >= 0 && b2h[rank[s]]){
          for (int k = rank[s]+1; !bh[k] && b2h[k]; k++) b2h[k] = false;
        }
      }
    }
    for (int i=0; i<n; ++i){
      pos[rank[i]] = i;
      bh[i] |= b2h[i];
    }
  }
  for (int i=0; i<n; ++i){
    rank[pos[i]] = i;
  }
}
// End of suffix array algorithm
 
 
// Begin of the O(n) longest common prefix algorithm
// Refer to "Linear-Time Longest-Common-Prefix Computation in Suffix
// Arrays and Its Applications" by Toru Kasai, Gunho Lee, Hiroki
// Arimura, Setsuo Arikawa, and Kunsoo Park.
int height[N];
// height[i] = length of the longest common prefix of suffix pos[i] and suffix pos[i-1]
// height[0] = 0
void getHeight(int n){
  for (int i=0; i<n; ++i) rank[pos[i]] = i;
  height[0] = 0;
  for (int i=0, h=0; i<n; ++i){
    if (rank[i] > 0){
      int j = pos[rank[i]-1];
      while (i + h < n && j + h < n && str[i+h] == str[j+h]) h++;
      height[rank[i]] = h;
      if (h > 0) h--;
    }
  }
}
// End of longest common prefixes algorithm
#-----------------------------------------------------------------#
##########################  #################################

#define MAXN 100000
/////////////////////////SUFFIX UlTRA FAST /////////////////////
char a[MAXN],b[MAXN];
char s[MAXN];
int suf[MAXN],lcp[MAXN];
int n;
void counting(int* a,int *b,int *val,int n,int k){
        int* c=new int[1+k];
        memset(c,0,sizeof(int)*(1+k));
        for(int i=0;i<n;++i)c[val[a[i]]]++;
        for(int i=1;i<=k;++i)c[i]+=c[i-1];
        for(int i=n-1;i>=0;--i)b[--c[val[a[i]]]]=a[i];
        delete[] c;
}
 
void suffix(int* val,int* suf,int n,int k){
        val[n]=val[n+1]=val[n+2]=0;
        int n0=(n+2)/3,n2=n/3,n02=n0+n2;
        int *val12=new int[n02+3];
        int *suf12=new int[n02+3];
        for(int i=0,j=0;i<n+(n%3==1);++i)if(i%3)val12[j++]=i;
        counting(val12,suf12,val+2,n02,k);
        counting(suf12,val12,val+1,n02,k);
        counting(val12,suf12,val,n02,k);
        int a=-1,b=-1,c=-1,ct=0;
        for(int i=0;i<n02;++i){
                int ind=suf12[i];
                if(val[ind]!=a or val[ind+1]!=b or val[ind+2]!=c)++ct;
                val12[(ind%3==1)?ind/3:ind/3+n0]=ct;
                a=val[ind],b=val[ind+1],c=val[ind+2];
        }
        if(ct==n02){
                val12[n02]=0;
                for(int i=0;i<n02;++i){
                        suf12[val12[i]-1]=i;
                }
        }
        else{
                suffix(val12,suf12,n02,ct);
                for(int i=0;i<n02;++i){
                        int ind=suf12[i];
                        val12[ind]=i+1;
                }
        }
        int *val0=new int[n0];
        int *suf0=new int[n0];
        for(int i=0,j=0;i<n02;++i){
                int ind=suf12[i];
                if(ind<n0)val0[j++]=ind*3;
        }
        counting(val0,suf0,val,n0,k);
        a=0,b=(n%3==1),c=0;
        #define comp1(a1,b1,a2,b2) (((a1)!=(a2))?((a1)<(a2)):((b1)<(b2)))
        #define comp2(a1,b1,c1,a2,b2,c2) (((a1)!=(a2))?((a1)<(a2)):(((b1)!=(b2))?((b1)<(b2)):((c1)<(c2))))
        #define get(i) ((i)<n0?(3*i+1):(3*(i-n0)+2))
        while(a<n0&&b<n02){
                int ind0=suf0[a],ind12=get(suf12[b]);
                if(ind12%3==1?comp1(val[ind0],val12[ind0/3],val[ind12],val12[suf12[b]+n0]):
                comp2(val[ind0],val[ind0+1],val12[ind0/3+n0],val[ind12],val[ind12+1],val12[suf12[b]-n0+1]))suf[c++]=ind0,a++;
                else suf[c++]=ind12,b++;
        }
        while(a<n0)suf[c++]=suf0[a++];
        while(b<n02)suf[c++]=get(suf12[b]),b++;
        delete [] val12; delete [] suf12; delete [] val0; delete [] suf0;
}
void lcp1(int *lcp,int *pos,int n){
     int *rank = new int[n+3];
     for(int i=0;i<n;++i)rank[pos[i]]=i;
     	int h=0;
     	for(int i=0;i<n;++i)if(rank[i]){
     		int j=pos[rank[i]-1];
     		while(s[i+h]==s[j+h])++h;
     		lcp[rank[i]-1]=h;
     		if(h)h--;
     	}
     lcp[n-1] = 0;
     delete [] rank;
} 
void _suffix(){
        int *val=new int[n+3];
        for(int i=0;i<n;++i)val[i]=s[i];
        suffix(val,suf,n,127);     
} 
void _lcp(){
     lcp1(lcp,suf,n);
}
/////////////////////////SUFFIX UlTRA FAST /////////////////////
/// LCP(i) = Longest Common Preffix ( i , i + 1 )  0 <= i <= n - 2 
int main(){
     gets(s):
     n=strlen(s),        
     _suffix();
     _lcp();
    
     for(int i=0;i<n;++i)printf("%d\n",suf[i]);
     puts("**");
     for(int i=0;i<n;++i)printf("%d\n",lcp[i]);        

}
/// LCP(i) = Longest Common Preffix ( i , i + 1 )  0 <= i <= n - 2 
int main(){
     gets(a);
     m = strlen(a);
     gets(b);
     strcat(s,a);
     strcat(s,"$");
     strcat(s,b);
     n=strlen(s),        
     _suffix();
     _lcp();
     /*//TEST
     for(int i=0;i<n;++i)printf("%d\n",suf[i]);

     puts("**");

     for(int i=0;i<n;++i)printf("%d\n",lcp[i]);        */
     int maxi = 0;
     for( int i = 0 ; i < n ; ++i )
          if( ( suf[i] < m && suf[i+1] > m )|| ( suf[i+1] < m && suf[i] > m ))
               maxi = max(maxi,lcp[i]);
     printf("%d\n",maxi);
}
//
##########################  Suffix array #################################
#define MAXN 1000005
int n,t;  //n es el tama\F1o de la cadena
int p[MAXN],r[MAXN],h[MAXN];
//p es el inverso del suffix array, no usa indices del suffix array ordenado
//h el el tama\F1o del lcp entre el i-esimo y el i+1-esimo elemento de suffix array ordenado
string s;
void fix_index(int *b, int *e) {
   int pkm1, pk, np, i, d, m;
   pkm1 = p[*b + t];
   m = e - b; d = 0;
   np = b - r;
   for(i = 0; i < m; i++) {
      if (((pk = p[*b+t]) != pkm1) && !(np <= pkm1 && pk < np+m)) {
         pkm1 = pk;
         d = i;
      }
      p[*(b++)] = np + d;
   }
}
bool comp(int i, int j) {
   return p[i + t] < p[j + t];
}
void suff_arr() {
   int i, j, bc[256];
   t = 1;
   for(i = 0; i < 256; i++) bc[i] = 0;  //alfabeto
   for(i = 0; i < n; i++) ++bc[int(s[i])]; //counting sort inicial del alfabeto
   for(i = 1; i < 256; i++) bc[i] += bc[i - 1];
   for(i = 0; i < n; i++) r[--bc[int(s[i])]] = i;
   for(i = n - 1; i >= 0; i--) p[i] = bc[int(s[i])];
   for(t = 1; t < n; t *= 2) {
      for(i = 0, j = 1; i < n; i = j++) {
         while(j < n && p[r[j]] == p[r[i]]) ++j;
         if (j - i > 1) {
            sort(r + i, r + j, comp);
            fix_index(r + i, r + j);
         }
      }
   }
}
void lcp() {
   int tam = 0, i, j;
   for(i = 0; i < n; i++)if (p[i] > 0) {
      j = r[p[i] - 1];
      while(s[i + tam] == s[j + tam]) ++tam;
      h[p[i] - 1] = tam;
      if (tam > 0) --tam;
   }
   h[n - 1] = 0;
}

int main(){
   s="margarita$";
   n=s.size();
   suff_arr();
   lcp();
   for(int i=0;i<n;i++)cout<<r[i]<<" ";cout<<endl;
   for(int i=0;i<n;i++)cout<<h[i]<<" ";cout<<endl;
   return 0;
}
### Suffix Array O(nlogn) ################################################################
//http://www.spoj.pl/problems/SARRAY/
//http://www.spoj.pl/problems/SUBST1/

#define checkMod(i, n) (((i) < (n)) ? (i) : (i) - (n))
#define MAXN 1000000
#define ALPH_SIZE 256

using namespace std;

char s[MAXN + 5];
int n;

int p[MAXN + 1], lcp[MAXN + 1], cnt[MAXN + 1], c[MAXN + 1];
int pn[MAXN + 1], cn[MAXN + 1];

void build_suffix_array()
{
	memset(cnt, 0, ALPH_SIZE * sizeof(int));
	for(int i=0; i<n; ++i) ++cnt[s[i]];
	for(int i=1; i<ALPH_SIZE; ++i) cnt[i] += cnt[i-1];
	for(int i=0; i<n; ++i) p[--cnt[s[i]]] = i;
	
	c[p[0]] = 0;
	int classes = 1;
	for(int i=1; i<n; ++i){
		if(s[p[i]] != s[p[i-1]]) ++classes;
		c[p[i]] = classes-1;
	}
	
	for(int h=0; (1<<h)<n; ++h){
		for(int i=0; i<n; ++i) pn[i] = checkMod(p[i] - (1<<h) + n, n);
		
		memset(cnt, 0, classes * sizeof(int));
		for(int i=0; i<n; ++i) ++cnt[c[pn[i]]];
		for(int i=1; i<classes; ++i) cnt[i] += cnt[i-1];
		for(int i=n-1; i>=0; i--) p[--cnt[c[pn[i]]]] = pn[i];
		
		for(int i=0; i<n; ++i) pn[i] = checkMod(p[i] + (1<<h), n);
		
		cn[p[0]] = 0;
		classes = 1;
		for(int i=1; i<n; ++i){
			if(c[p[i]] != c[p[i-1]] || c[pn[i]] != c[pn[i-1]]) ++classes;
			cn[p[i]] = classes-1;
		}
		memcpy(c, cn, n * sizeof(int));
	}
}

void build_lcp() {
	int k = 0;
	for(int i = 0; i < n; i++) if (c[i]) {
		int j = p[c[i] - 1];
		while(s[i + k] == s[j + k]) k++;
		lcp[c[i] - 1] = k;
		if (k) k--;
	}
	lcp[n - 1] = 0;
}
#-----------------------------------------------------------------#

########################## StringUtilities( Chen ) #################################
#include <iomanip>
#include <ctime>
#include <numeric>
#include <functional>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <climits>
#include <cmath>
#include <cctype>
#include <sstream>
#include <map>
#include <set>
#include <cstdio>
#include <queue>
#define f(i,x,y) for(int i=x;i<y;i++)
#define fd(i,y,x) for(int i=y;i>=x;i--)
#define FOR(it,A) for( typeof A.begin() it = A.begin(); it!=A.end(); it++)
#define impr(A) for( typeof A.begin() chen = A.begin(); chen !=A.end(); chen++ ) cout<<*chen<<" "; cout<<endl
#define ll long long
#define vint vector<int>
#define clr(A,x) memset(A,x,sizeof(A))
#define CLR(v) f(i,0,n) v[i].clear()
#define oo (1<<30)
#define ones(x) __builtin_popcount(x)
#define all(v) (v).begin(),(v).end()
#define rall(v) (v).rbegin(),(v).rend()
#define pb push_back
#define eps (1e-9)
#define cua(x) (x)*(x)
using namespace std;
/*

int KMP(char *S, char *K){
	int n = strlen(S), m = strlen(K);
	int j = T[0] = -1;
	f(i,1,m+1){
		while( j!=-1 && K[i-1]!=K[j] ) j = T[j];
		T[i] = ++j;
	}

	j = 0;
	f(i,1,n+1){
		while( j!=-1 && (S[i-1]!=K[j]) ) j = T[j];
		++j;
		if( j==m ) return 1;
	}
	return 0;
}
*/
int next[1000];
int lcp[1000];
void get_next(char *st,int n){
	int p=-1,t;
	f(i,1,n){
		if( p>i && next[i-t]<p-i ) next[i] = next[i-t];
		else{
			int j = max(0,p-i);
			for(;i+j<n;j++)if( st[i+j]!=st[j] ) break;
			next[i]=j, t=i, p=i+next[i];
		}
	}
}
void kmpext(char *a,char *b,int n,int m){
	int p=-1,t;
	get_next(b,m);
	f(i,0,n){
		if( p>i && next[i-t]<p-i ) lcp[i] = next[i-t];
		else{
			int j = max(0,p-i);
			for(;i+j<n;j++)if( a[i+j]!=b[j] ) break;
			lcp[i]=j, t=i, p=i+lcp[i];
		}
	}
}
// Find the index of the Minimun Lexicografical Rotation (Shift) in a String O(n)
// 4929 ACM
int shift(char *s,int n){
	int i = 0, j = 1, k = 0;
	char a,b;
	while( j<n && i+k+1<n ){
		a = s[i+k]; b = s[(j+k)%n];
		if( a==b ) k++;
		else if( a<b ) j=j+k+1, k = 0;
		else i = max(i+k+1,j), j = i+1, k = 0;
	}
	return i;
}
############################# SHIFT ##############################

// Find the index of the Minimun Lexicografical Rotation (Shift) in a String O(n)

int shift( string s ) {
	int i = 0, j = 1, k = 0 , ns = s.size();
	char a,b;
	while( j<ns && i+k+1<ns ){
		a = s[i+k]; b = s[(j+k)%ns];
		if( a==b ) k++;
		else if( a<b ) j=j+k+1, k = 0;
		else i = max(i+k+1,j), j = i+1, k = 0;
	}
	return i;
}
############################# SHIFT  duval_algorithm ##############################
// by EMAXX
string shift (string s) {
	s += s;
	int n = (int) s.length();
	int i = 0, ans = 0;
	while (i < n/2) {
		ans = i;
		int j = i + 1 , k = i;
		while (j < n && s[k] <= s[j]) {
			if (s[k] < s[j])
				k = i;
			else
				++k;
			++j;
		}
		while (i <= k)  i += j - k;
	}
	return s.substr (ans, n/2);
}
//By diego
string minlex(string s){
	
	int n = s.size();
s = s + s;
   int mini = 0, p = 1, l = 0;
   while(p < n && mini + l + 1 < n)
      if(s[mini + l] == s[p + l])
         l++;
      else if(s[mini + l] < s[p + l]){
         p = p + l + 1;
         l = 0;
      }else if(s[mini + l] > s[p + l]){
         mini = max(mini + l + 1, p);
         p = mini + 1;
         l = 0;
      }
   s = s.substr(mini, n);
   return s;
}

#-----------------------------------------------------------------#
int main()
{
	string cad , cad2; int n;
	while( cin>>cad >> cad2 ){
		kmpext( &cad[0], &cad2[0] ,cad.size() , cad2.size() );
		f(i,0,cad.size())cout<<lcp[i]<<" ";cout<<endl;
	}
}

#-----------------------------------------------------------------#

########################## Z ALGORITHM #################################
//Given a string S of length n, the Z Algorithm produces an array Z where Z[i] is the length 
//of the longest substring starting from S[i] which is also a prefix of S
// EMAXX
// Por defecto z[ 0 ] = 0;
vector<int> z_function (string s) {
	int n = (int) s.length();
	vector<int> z (n);
	for (int i=1, l=0, r=0; i<n; ++i) {
		if (i <= r)
			z[i] = min (r-i+1, z[i-l]);
		while (i+z[i] < n && s[z[i]] == s[i+z[i]])
			++z[i];
		if (i+z[i]-1 > r)
			l = i,  r = i+z[i]-1;
	}
	return z;
}

int L = 0, R = 0;
for (int i = 1; i < n; i++) {
  if (i > R) {
    L = R = i;
    while (R < n && s[R-L] == s[R]) R++;
    z[i] = R-L; R--;
  } else {
    int k = i-L;
    if (z[k] < R-i+1) z[i] = z[k];
    else {
      L = i;
      while (R < n && s[R-L] == s[R]) R++;
      z[i] = R-L; R--;
    }
  }
  z[0] = n;
}
#-----------------------------------------------------------------#
fz[0] = L;

for(int i = 1,l = 0,r = 0;i < L;++i){
    if(l && i + fz[i - l] < r) fz[i] = fz[i - l];
    else{
        int j = 0;
        if(l) j = min(fz[i - l],r - i);
        j = max(j,0);

        while(i + j < L && s[i + j] == s[j]) ++j;

        fz[i] = j;
        l = i;
        r = i + j;
    }
}

###################################################################
################# MATRICES #######################################
###################################################################

############################# FAST MATRIX EXPONENTIATION #############################
//Codeforces Round #230 (Div. 1)  C. Yet Another Number Sequence

//UVA 10229 - Modular Fibonacci
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second
 
#define N 1000
#define MAXE 1100000
#define MAXV 2000

#define INF (1<<28)

#define test puts( "******************test**********************" );

typedef long long ll;
typedef vector< int > vi;

ll MOD;

struct Matrix{
	ll M[ 2 ][ 2 ];
	Matrix(){
		M[ 0 ][ 0 ] = M[ 0 ][ 1 ] = M[ 1 ][ 0 ] = 1;
		M[ 1 ][ 1 ] = 0;
	}
	Matrix( ll t ){
		M[ 0 ][ 0 ] = M[ 1 ][ 1 ] = 1;
		M[ 0 ][ 1 ] = M[ 1 ][ 0 ] = 0;
	}
};
Matrix operator * ( const Matrix &a , const Matrix &b ){
	Matrix C;
	REP( i , 2 )REP( j , 2 ){
		ll dev = 0;
		REP( k , 2 ) dev = ( dev + a.M[ i ][ k ] * b.M[ k ][ j ] );
		C.M[ i ][ j ] = dev%MOD;
	}
	return C;
}
Matrix pow( Matrix a , ll b ){
	Matrix ans( 1 );
	while( b ){
		if( b & 1 ) ans = ans * a;
		a = a * a;
		b >>= 1 ;
	}
	return ans;
}
int main(){
	ll n , t;
	while( cin >> n >> t ){
		MOD = (1 << t);
		if( n == 0 ) cout << 0 << '\n';
		else{
			Matrix mat;
			mat = pow( mat , n - 1 );
			cout << mat.M[ 0 ][ 0 ] << '\n';
		}
	}
}

#-----------------------------------------------------------------#
########################## GAUSSIAN ELIMINATION #################################
//inverts and find determinant of matrix O( n^3 )
// Fidel TC UNLP 2013 
// http://www.youtube.com/watch?v=j4skKKJ_4bw
//http://en.wikipedia.org/wiki/Gaussian_elimination

/* Computing determinants
To explain how Gaussian elimination allows to compute the determinant of a square matrix, we have to recall how the elementary row operations change the determinant:
 - Swapping two rows multiplies the determinant by -1
 - Multiplying a row by a nonzero scalar multiplies the determinant by the same scalar
 - Adding to one row a scalar multiple of another does not change the determinant.
*/
// UVA 684 - Integral Determinant
// Aplication Polynomial_interpolation( vandermonde matrix )
// http://en.wikipedia.org/wiki/Polynomial_interpolation#Uniqueness_of_the_interpolating_polynomial
#define N 35
#define EPS (1e-8)
double det;
bool invert( double A[ N ][ N ] , double B[ N ][ N ] , int n )
{
	det = 1;
	REP( i , n )
	{
		int jmax = i ;
		for( int j = i + 1 ; j < n ; ++j )
			if( abs( A[ j ][ i ] ) > abs( A[ jmax ][ i ] ) ) jmax = j;
		if( jmax > i ) det = -det;
		REP( j , n )
			swap( A[ i ][ j ] , A[ jmax ][ j ] ) , swap( B[ i ][ j ] , B[ jmax ][ j ] );
		if( abs( A[ i ][ i ] ) < EPS ) {
			det = 0;
			return 0;
		}
		double tmp = A[ i ][ i ];
		det *= tmp;
		REP( j , n ) A[ i ][ j ] /= tmp , B[ i ][ j ] /= tmp;
		REP( j , n )
		{
			if( i == j )continue;
			tmp = A[ j ][ i ];
			REP( k , n )
				A[ j ][ k ] -= A[ i ][ k ]*tmp , B[ j ][ k ] -= B[ i ][ k ]*tmp;
		}
	}
	return 1;
}
printf( "%d\n" , int( floor( det + 0.1 ) ) );

####################### Gaussian Eliminaton MOD ####################################
#define MAX_R 500
#define MAX_C 501
int R,C,MOD;

struct Matrix{
    int X[MAX_R][MAX_C];
    Matrix(){}
};

//cuidado con overflow
int exp(int a, int n){
    if(n==0) return 1;
    if(n==1) return a;
    
    int aux=exp(a,n/2); 
    if(n&1) return ((long long)a*(aux*aux)%MOD)%MOD;
    return (aux*aux)%MOD;
}

void GaussianElimination(Matrix &M0){
    for(int i = 0,r = 0;r<R && i<C;++i){
        bool found = false;
        
        for(int j = r;j<R;++j){
            if(M0.X[j][i]>0){
                found = true;
                
                if(j==r) break;
                
                for(int k = i;k<C;++k) swap(M0.X[r][k],M0.X[j][k]);
                break;
            }
        }
        
        if(found){
            int aux = modular_inverse(M0.X[r][i],MOD);
            
            for(int j = i;j<C;++j) M0.X[r][j] = (M0.X[r][j]*aux)%MOD;
            
            for(int j = r+1;j<R;++j){
                aux = MOD-M0.X[j][i];
                for(int k = i;k<C;++k)
                    M0.X[j][k] = (M0.X[j][k]+aux*M0.X[r][k])%MOD;
            }
            
            ++r;
        }else return;
    }
    
    for(int i = R-1;i>0;--i)
        for(int j = 0;j<i;++j)
            M0.X[j][C-1] = (M0.X[j][C-1]+(MOD-M0.X[j][i])*M0.X[i][C-1])%MOD;
}
#-----------------------------------------------------------------#
//matrix rank mod 2 
int Rank(vector<vector<int> > A){
	if (A.size() == 0) return 0;
	int n = A.size(), m = A[0].size();
	int row = 0;
	f(c, 0, m)
	{
		if (row == n) return n;
		f(j, row + 1, n) if (A[j][c] == 1)
		{
			swap(A[row], A[j]);
			break;
		}
		if (A[row][c] == 0)
			continue;

		f(j, row + 1, n) if (A[j][c])
		{
			f(k, c, m) 
				A[j][k] ^= A[row][k];
		}
		row ++;
	}
	return row;
}
///AJI gauss
typedef long double tipo;
typedef vector<tipo> Vec;
typedef vector<Vec> Mat;

#define forn(i,n) for(int i=0;i<(int)(n);i++)
#define forsn(i,s,n) for(int i=(s);i<(int)(n);i++)
#define dforn(i,n) for(int i=(n)-1;i>=0;i--)

typedef vector<tipo> Vec;
typedef vector<Vec> Mat;
#define eps 1e-10
#define feq(a, b) (fabs(a-b)<eps)
bool resolver_ev(Mat a, Vec y, Vec &x, Mat &ev){
	int n = a.size(), m = n?a[0].size():0, rw = min(n, m);
	vector<int> p; forn(i,m) p.push_back(i);
	forn(i, rw){
		int uc=i, uf=i;
		// aca pivotea. lo unico importante es que a[i][i] sea no nulo
		forsn(f, i, n) forsn(c, i, m) if(fabs(a[f][c])>fabs(a[uf][uc])) {uf=f;uc=c;}
		if (feq(a[uf][uc], 0)) { rw = i; break; }
		forn(j, n) swap(a[j][i], a[j][uc]);

		swap(a[i], a[uf]); swap(y[i], y[uf]); swap(p[i], p[uc]);
		// fin pivoteo
		tipo inv = 1 / a[i][i]; //aca divide
		forsn(j, i+1, n) {
			tipo v = a[j][i] * inv;
			forsn(k, i, m) a[j][k]-=v * a[i][k];
			y[j] -= v*y[i];
		}
	} // rw = rango(a), aca la matriz esta triangulada
	forsn(i, rw, n) if (!feq(y[i],0)) return false; // checkeo de compatibilidad
	x = vector<tipo>(m, 0);
	dforn(i, rw){
		tipo s = y[i];
		forsn(j, i+1, rw) s -= a[i][j]*x[p[j]];
		x[p[i]] = s / a[i][i]; //aca divide
	}
	ev = Mat(m-rw, Vec(m, 0)); // Esta parte va SOLO si se necesita el ev
	forn(k, m-rw) {
		ev[k][p[k+rw]] = 1;
		dforn(i, rw){
			tipo s = -a[i][k+rw];
			forsn(j, i+1, rw) s -= a[i][j]*ev[k][p[j]];
			ev[k][p[i]] = s / a[i][i]; //aca divide
		}
	}
	return true;
}
bool diagonalizar(Mat &a){
	// PRE: a.cols > a.filas
	// PRE: las primeras (a.filas) columnas de a son l.i.
	int n = a.size(), m = a[0].size();
	forn(i, n){
		int uf = i;
		forsn(k, i, n) if (fabs(a[k][i]) > fabs(a[uf][i])) uf = k;
		if (feq(a[uf][i], 0)) return false;
		swap(a[i], a[uf]);
		tipo inv = 1 / a[i][i]; // aca divide
		forn(j, n) if (j != i) {
			tipo v = a[j][i] * inv;
			forsn(k, i, m) a[j][k] -= v * a[i][k];
		}
		forsn(k, i, m) a[i][k] *= inv;
	}
	return true;
}
///SIMPLEX
//TIMUS 1764. Transsib
#include <bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; i++ )
#define clr( t , val ) memset( t , val , sizeof(t) )

#define all(v)  v.begin() , v.end()
#define pb push_back
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define test() cerr << "hola que hace ?" << endl;
#define DEBUG( x ) cerr <<  #x << "=" << x << endl;
#define DEBUG2( x , y ) cerr << #x << "=" << x << " " << #y << "=" << y << endl;

typedef long long ll;
typedef pair< int , int > pii;
typedef vector< int > vi;

/*
simplex::solve(A,b,c,x)
	maximize     c' * x
	subject to   A * x' <= b'

	input:
		A -- an m x n matrix
		b -- an m-dimensional vector
		c -- an n-dimensional vector
		x -- a vector where the optimal solution will be stored

	returns:
		value of the optimal solutions,
		or INF if unbounded or infeasible (?)
 */ 

#define INF				1e+10
#define EPS				1e-10
typedef double lf;
typedef vector<lf> Row;
typedef vector<Row> Mat;
typedef vector<int> vi;
inline int sign(lf x){return (x>+EPS)-(x<-EPS);}

namespace simplex
{
	const int MAXN=4;
	const int MAXM=6;
	int n, m;
	lf A[MAXM+1][MAXN+1];
	int basis[MAXM+1],out[MAXN+1];

	void pivot(int a,int b) {
		for(int i=0;i<=m;++i)if(i!=a&&sign(A[i][b]))
		for(int j=0;j<=n;++j)if(j!=b)A[i][j]-=A[a][j]*A[i][b]/A[a][b];
		for(int j=0;j<=n;++j)if(j!=b)A[a][j]/=A[a][b];
		for(int i=0;i<=m;++i)if(i!=a)A[i][b]/=-A[a][b];
		A[a][b]=1/A[a][b];
		swap(basis[a],out[b]);
	}

	lf simplex(Row& x) {
		int i,j,ii,jj;
		for(i=0;i<=m;++i)basis[i]=-i;
		for(j=0;j<=n;++j)out[j]=j;
		while(1) {
			ii=1,jj=0;
			for(i=1;i<=m;++i)
				if(mp(A[i][n],basis[i])<mp(A[ii][n],basis[ii]))ii=i;
			if(A[ii][n]>=0)break;
			for(j=0;j<n;++j)if(A[ii][j]<A[ii][jj])jj=j;
			if(A[ii][jj]>=0)return-INF;
			pivot(ii,jj);
		}
		while(1) {
			ii=1,jj=0;
			for(j=0;j<n;++j)if(mp(A[0][j],out[j])<mp(A[0][jj],out[jj]))jj=j;
			if(A[0][jj]>=0)break;
			for(i=1;i<=m;++i)
				if (A[i][jj]>0&&(A[ii][jj]<=0
					||mp(A[i][n]/A[i][jj],basis[i])
					<mp(A[ii][n]/A[ii][jj],basis[ii]))) ii=i;
			if(A[ii][jj]<=0)return+INF;
			pivot(ii,jj);
		}
		x.resize(n);
		for(j=0;j<n;++j)x[j]=0;
		for(i=1;i<=m;++i)if(basis[i]>=0)x[basis[i]]=A[i][n];
		return A[0][n];
	}

	lf solve ( const Mat& _A, const Row& b, const Row& c, Row& x ) {
		n = c.size();
		m = b.size();
		for ( int i = 0; i < n; ++i )
			A[0][i] = -c[i];
		A[0][n] = 0;
		for ( int i = 0; i < m; ++i ) {
			for ( int j = 0; j < n; ++j )
				A[i+1][j] = _A[i][j];
			A[i+1][n]=b[i];
		}
		return simplex(x);
	}
}

int main(){

	const int m = 6;
	const int n = 4;
	
	Row b( m );
	REP( i , m ) {
		int x;
		sc( x );
		b[ i ] = x;
	}
	
	Mat A ( m, Row ( n , 0 ) );
	
	A[ 0 ][ 0 ] = A[ 0 ][ 3 ] = 1;
	A[ 1 ][ 0 ] = A[ 1 ][ 2 ] = 1;
	A[ 2 ][ 2 ] = A[ 2 ][ 3 ] = 1;
	A[ 3 ][ 1 ] = A[ 3 ][ 2 ] = 1;
	A[ 4 ][ 1 ] = A[ 4 ][ 3 ] = 1;
	A[ 5 ][ 0 ] = A[ 5 ][ 1 ] = 1;

	Row c( n , 1 ) , x;
	simplex::solve( A , b , c , x );
	REP( i , n ) printf( "%.3lf%c" , x[ i ] , (i + 1 == n) ? 10 : 32 );
	
}


###################################################################
################# DATA STRUCTURES #######################################
###################################################################

######################### SEGMENT TREE #################################
//ACM 2191 - Potentiometers
// Sumas
// Soporta querys de intervalos y update de un solo elemento 
#define N 200005
#define NEUTRAL 0
#define v1 ( ( node << 1 ) + 1 )
#define v2 ( v1 + 1 )
#define med ( (a + b) >> 1 )
#define LEFT v1 , a , med
#define RIGHT v2 , med + 1 , b

int A[ N ];
int T[ 4*N ];

void build_tree( int node , int a , int b ){
	if( a == b ){
		T[ node ] = A[ a ];
		return;
	}
	build_tree( LEFT );build_tree( RIGHT );
	T[ node ] = T[ v1 ] + T[ v2 ];
}
void update( int node , int a , int b , int x , int val ){
	if( x > b || a > x ) return;
	if( a == b ){
		T[ node ] = val;
		return;
	}
	update( LEFT , x , val );update( RIGHT , x , val );
	T[ node ] = T[ v1 ] + T[ v2 ];
}
int query( int node , int a ,  int b , int lo , int hi ){
	if( lo > b || a > hi ) return NEUTRAL;
	if( a >= lo && hi >= b ) return T[ node ];
	return query( LEFT , lo , hi ) + query( RIGHT , lo , hi );	
}

// Version que soporta operaciones max , best_subarray_sum ( build_tree y query )
//( SPOJ "GSS1" 1043. Can you answer these queries I )
// ( SPOJ "GSS3" 1716. Can you answer these queries III )
#define N 50005 
#define INF (1<<29)

#define v1 ( ( node << 1) + 1 )
#define v2 ( v1 + 1 )
#define med ( ( a + b ) >> 1 )
#define LEFT v1 , a , med
#define RIGHT v2 , med + 1 , b

struct Node{
	int best , der , izq , sum;
	Node(){
		sum = 0;
		izq = der = best = -INF;
	}
	Node( int val ): best( val ) , der( val ) , izq( val ) , sum( val ) {};
} T[ 4*N ] , A[ N ] , NEUTRAL;
Node operator +( const Node &a , const Node &b ){
	Node ans;
	ans.sum = a.sum + b.sum;
	ans.der = max( b.der , b.sum + a.der );
	ans.izq = max( a.izq , a.sum + b.izq );
	ans.best = max( a.best , b.best );
	ans.best = max( ans.best , a.der + b.izq );
	return ans;
}
void build_tree( int node , int a , int b ){
	if( a == b ){
		T[ node ] = Node( A[ a ] );
		return;
	}
	build_tree( LEFT );build_tree( RIGHT );
	T[ node ] = T[ v1 ] + T[ v2 ];
}
void update( int node , int a , int b , int x , int val ){
	if( x > b || a > x ) return;
	if( a == b ){
		T[ node ] = Node( val );
		return;
	}
	update( LEFT , x , val );update( RIGHT , x , val );
	T[ node ] = T[ v1 ] + T[ v2 ];
}
Node query( int node , int a , int b , int lo , int hi ){
	if( lo > b || a > hi ) return NEUTRAL;
	if( a >= lo && hi >= b ) return T[ node ];
	return query( LEFT , lo , hi ) + query( RIGHT , lo , hi );
}

// LAZY PROPAGATION
// SPOJ 8002. Horrible Queries
// Ojo sol usa operaciones de SUMA 
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define N 100000

typedef vector< int > vi;
typedef long long ll;

#define v1 ((node<<1)+1)
#define v2 (v1+1)
#define med ((a+b)>>1)
#define LEFT v1 , a , med
#define RIGHT v2 , med + 1 , b

ll T[ 4*N + 5 ] , flag[ 4*N + 5 ];
void push( int node , int a , int b ){
	if( !flag[ node ] ) return;
	T[ node ] += flag[ node ] * ( b - a + 1LL );
	if( a != b ){
		flag[ v1 ] += flag[ node ];
		flag[ v2 ] += flag[ node ];
	}
	flag[ node ] = 0;
}
ll query( int node , int a , int b , int lo , int hi ){
	push( node , a , b );
	if( lo > b || a > hi ) return 0;
	if( a >= lo && hi >= b ) return T[ node ];
	return query( LEFT , lo , hi ) + query( RIGHT , lo , hi );
}
void update( int node , int a , int b , int lo , int hi , int val ){
	push( node , a , b );
	if( lo > b || a > hi ) return;
	if( a >= lo && hi >= b ) {
		flag[ node ] = val;
		push( node , a , b );
		return;
	}
	update( LEFT , lo , hi , val );
	update( RIGHT , lo , hi , val );
	T[ node ] = T[ v1 ] + T[ v2 ];
}
int main(){
	int cases , n , Q , op , lo , hi , val;
	sc( cases );
	REP( tc , cases ){
		sc( n ) , sc( Q );
		clr( T , 0 ) , clr( flag , 0 );
		REP( i , Q ){
			sc( op );
			if( op == 0 ){
				sc( lo ) , sc( hi ) , sc( val );
				lo -- , hi --;
				update( 0 , 0 , n - 1 , lo , hi , val );
			}else{
				sc( lo ) , sc( hi );
				lo -- , hi --;
				printf( "%lld\n" , query( 0 , 0 , n - 1 , lo , hi ) );
			}
		}
	}
}
//Aplication
//CODEFORCES Croc Champ 2013 - Round 1  E. Copying Data	

// SEGMENT TREE 2D max y min
//http://e-maxx.ru/algo/segment_tree
pii T[ 4*N ][ 4*N ];
int n , m ;
pii g( pii a , pii b ){
	return mp( max( a.fi , b.fi ) , min( a.se , b.se ) );
}
pii QueryY( int nodex , int nodey , int ay , int by , int ylo , int yhi )
{
	if( ay > yhi || ylo > by ) return mp( -INF , INF );
	if( ay >= ylo && yhi >= by ) return T[ nodex ][ nodey ];
	int v1 = 2*nodey + 1 , v2 = v1 + 1 , med = ( ay + by )/2;
	return g( QueryY( nodex , v1 , ay , med , ylo , yhi ) , QueryY( nodex , v2 , med + 1 , by , ylo , yhi ) );
}

pii QueryX( int nodex , int ax , int bx , int xlo , int xhi , int ylo , int yhi )
{
	if( ax > xhi || xlo > bx ) return mp( -INF , INF );
	if( ax >= xlo && xhi >= bx ) return QueryY( nodex , 0 , 0 , m - 1 , ylo , yhi );
	int v1 = 2*nodex + 1 , v2 = v1 + 1 , med = ( ax + bx )/2;
	return g( QueryX( v1 , ax , med , xlo , xhi , ylo , yhi ) , QueryX( v2 , med + 1 , bx , xlo , xhi , ylo , yhi ) );
}

void updateY( int nodex , int ax , int bx , int nodey , int ay , int by , int x , int y , int val )
{
	if( ay > y || y > by ) return;
	if( ay == by )
	{
		if( ax == bx )
			T[ nodex ][ nodey ] = mp( val , val );
		else T[ nodex ][ nodey ] = g( T[ 2*nodex + 1 ][ nodey ] , T[ 2*nodex + 2 ][ nodey ] );
		return;
	}
	int v1 = 2*nodey + 1 , v2 = v1 + 1 , med =( ay + by )/2;
	updateY( nodex , ax , bx , v1 , ay , med , x , y , val );
	updateY( nodex , ax , bx , v2 , med + 1 , by , x , y , val );
	T[ nodex ][ nodey ] = g( T[ nodex ][ v1 ] , T[ nodex ][ v2 ] );
}
void updateX( int nodex , int ax , int bx , int x , int y , int val )
{
	if( ax > x || x > bx ) return;
	if( ax == bx )
	{
		updateY( nodex , ax , bx , 0 , 0 , m - 1 , x , y , val );
		return;
	}
	int v1 = 2*nodex + 1 , v2 = v1 + 1 , med = ( ax + bx )/2;
	updateX( v1 , ax , med , x , y , val );
	updateX( v2 , med + 1 , bx , x , y , val );
	updateY( nodex , ax , bx , 0 , 0 , m - 1 , x , y , val );
}
pii q = QueryX( 0 , 0 , n - 1 , xlo , xhi , ylo , yhi );
updateX( 0 , 0 , n - 1 , x , y , val );
///Lazy propagation with pointers + hashing
//321 CDF div2 E. Kefa and Watch
const int dim = 3;
vll MOD = { 1000000007LL , 1000000009LL , 1000000021LL };
const ll B = 13;
struct Hash{
	ll h[ dim ];
	int len;
	Hash(){
		REP( k , dim ) h[ k ] = 0;
		len = 0;
	}
	Hash( int val ){
		REP( k , dim ) h[ k ] = val;
		len = 1;
	}
};
ll POT[ dim ][ N + 5 ];
ll AC[ dim ][ N + 5 ];

Hash operator + ( const Hash &h1 , const Hash &h2 ){
	Hash h;
	h.len = h1.len + h2.len;
	REP( k , dim )
		h.h[ k ] = (h1.h[ k ] * POT[ k ][ h2.len ] + h2.h[ k ]) % MOD[ k ];
	return h;
}
bool operator == ( const Hash &h1 , const Hash &h2 ){
	if( h1.len != h2.len ) return 0;
	REP( k , dim )
		if( h1.h[ k ] != h2.h[ k ] ) return 0;
	return 1;
}
struct Node{
	Hash h;
	int flag;
	Node * l , * r;
	Node(){ flag = -1;}
	Node( int val ) : l( NULL ) , r( NULL ) , flag( -1 ) , h( val ) {}
	Node( Node * l , Node * r ) : l( l ) , r( r ) , h() , flag( -1 ) {
		if( l ) h = h + (l->h);
		if( r ) h = h + (r->h);
	}
};
typedef Node * pNode;
#define mid ((a + b)>>1)
char S[ N + 5 ];
pNode build( int a , int b ){
	if( a == b ) return new Node( S[ a ] - '0' + 1 );
	return new Node( build( a , mid ) , build( mid + 1 , b ) );
}
Hash getHash( int len , int x ){
	Hash h;
	h.len = len;
	REP( k , dim ) h.h[ k ] = (AC[ k ][ len ] * (ll)x)%MOD[ k ];
	return h;
}
void push( pNode &t ){
	if( !t ) return;
	int &flag = (t -> flag);
	if( flag == -1 ) return;
	Hash &H = (t -> h);
	H = getHash( H.len , flag );
	if( (t -> l) ){
		(t -> l) -> flag = flag;
	}
	if( (t -> r) ){
		(t -> r) -> flag = flag;
	}
	flag = -1;
}
Hash get( pNode t , int a , int b , int lo , int hi ){
	push( t );
	if( lo > b || a > hi ) return Hash();
	if( lo <= a && b <= hi ) return (t -> h);
	return get( (t -> l) , a , mid , lo , hi ) + get( (t -> r) , mid + 1 , b , lo , hi );
}
void update( pNode t , int a , int b , int lo , int hi , int val ){
	push( t );
	if( lo > b || a > hi ) return;
	if( lo <= a && b <= hi ){
		(t -> flag) = val;
		push( t );
		return;
	}
	update( (t -> l) , a , mid , lo , hi , val );
	update( (t -> r) , mid + 1 , b , lo , hi , val );
	(t -> h) = ((t -> l) -> h) + ((t -> r) -> h);
}

int main(){
	REP( k , dim ){
		POT[ k ][ 0 ] = 1;
		for( int i = 1 ; i <= N ; ++i )
			POT[ k ][ i ] = (POT[ k ][ i - 1 ] * B)%MOD[ k ];
		for( int i = 0 ; i <= N ; ++i )
			AC[ k ][ i + 1 ] = (AC[ k ][ i ] + POT[ k ][ i ])%MOD[ k ];
	}

	int n , a , b;
	while( sc( n ) == 1 ){
		sc( a ) , sc( b );
		int m = a + b;
		scanf( "%s" , S );
		pNode t = build( 0 , n - 1 );
		
		Hash h = t->h;
		REP( i , m ){
			int op;
			sc( op );
			if( op == 1 ){
				int L , R , c;
				sc( L ) , sc( R ) , sc( c );
				L -- , R --;
				update( t , 0 , n - 1 , L , R , c + 1 );
			}else{
				int L , R , d;
				sc( L ) , sc( R ) , sc( d );
				L -- , R --;
				if( R - L + 1 == d ){
					puts( "YES" );
					continue;
				}
				Hash h1 = get( t , 0 , n - 1 , L , R - d );
				Hash h2 = get( t , 0 , n - 1 , L + d , R );
				puts( (h1 == h2) ? "YES" : "NO" );
			}
		}
	}
}
//persistent segment tree + pointers
//Codeforces Round #276 (Div. 1) E. Sign on Fence

#define mid ((a + b)>>1)

struct Info{
	int maxi , maxL , maxR , sum , len;
	Info(){ maxi = maxL = maxR = sum = len = 0;}
	Info( int val ){
		maxi = val;
		maxL = val;
		maxR = val;
		sum = val;
		len = 1;	
	}
};
bool operator == ( const Info &L , const Info &R ){
	return L.maxi == R.maxi && L.maxL == R.maxL && L.maxR == R.maxR && L.sum == R.sum && L.len == R.len;
}
Info operator + ( const Info &L , const Info &R ){
	if( L == Info() && R == Info() ) return Info();
	if( L == Info() ) return R;
	if( R == Info() ) return L;
	Info c;
	c.maxi = max( L.maxi , R.maxi );
	c.maxi = max( c.maxi , L.maxR + R.maxL );
	c.sum = L.sum + R.sum;
	c.len = L.len + R.len;
	c.maxL = L.maxL;
	c.maxR = R.maxR;
	if( L.sum == L.len ) c.maxL = max( c.maxL , L.sum + R.maxL );
	if( R.sum == R.len ) c.maxR = max( c.maxR , R.sum + L.maxR );
	return c;
}
struct Node {
	Node * l, * r;
	Info info;
	Node ( int val ) : l( NULL ) , r( NULL ) , info( val ) {}
	Node( Node * l , Node * r ) : l( l ) , r( r ){
		if( l ) info = info + (l -> info);
		if( r ) info = info + (r -> info);
	}
};
typedef Node * pNode;// puntero a node

pNode build( int a , int b ) {
	if( a == b ) return new Node( 0 );
	return new Node(
		build( a , mid ),
		build( mid + 1 , b )
	);
}
Info get( pNode t , int a , int b , int lo , int hi ){
	if( lo <= a && b <= hi ) return (t -> info);
	if( lo > b || a > hi ) return Info();
	return get( t->l , a , mid , lo , hi ) + get( t->r , mid + 1 , b , lo , hi );
}

pNode update( pNode t , int a , int b , int pos , int val ) {
	if( a == b ) return new Node( val );
	if( pos <= mid ){
		return new Node( update( t->l , a , mid , pos , val ) , t->r );
	}
	else{
		return new Node( t->l , update( t->r , mid + 1 , b , pos , val ) );
	}
}
int n;
bool f( int L , int R , int W , vector< pNode > &roots , int pos ){
	Info info = get( roots[ pos ] , 0 , n - 1 , L , R );
	return info.maxi >= W;
}
int binary( int L , int R , int W , vector< pNode > &roots ){
	int lo = 0 , hi = n;
	while( hi - lo > 1 ){
		int med = (lo + hi) >> 1;
		if( f( L , R , W , roots , med ) ) hi = med;
		else lo = med;
	}
	return hi;
}
int main(){
	
	while( sc( n ) == 1 ){
		vpii v;
		REP( i , n ){
			int x;
			sc( x );
			v.pb( mp( x , i ) );
		}
		sort( all( v ) );
		reverse( all( v ) );
		pNode p = build( 0 , n - 1 );

		vector< pNode > roots( 1 , p );
		REP( i , n ){
			int pos = v[ i ].se , val = v[ i ].fi;
			pNode pp = roots.back();
			roots.pb( update( pp , 0 , n - 1 , pos , 1 ) );
		}
		int Q;
		sc( Q );
		REP( q , Q ){
			int L , R , W;
			sc( L ) , sc( R ) , sc( W );
			L -- , R --;
			int pos = binary( L , R , W , roots );
			printf( "%d\n" , v[ pos - 1 ].fi );
		}
	}
}


#-------------------------------------------------------------------------------------------#
//KTH NUMBER
//#define N 100005
 
using namespace std;
 
int n , Q;
vector< int >T[4*N];
int A[N];
vector< int >v;
 
vector< int > pull( vector< int > &n1 , vector< int > &n2 )
{
      vector< int >ans(n1.size()+n2.size());
      merge( n1.begin() , n1.end() , n2.begin() , n2.end() , ans.begin() );
      return ans;      
}
void build_tree( int node , int a , int b )
{
      int v1 = 2*node + 1 , v2 = 2*node + 2 , med = ( a + b )/2;
      if( a == b )
      {
            T[ node ].push_back( A[a] );
            return ;      
      }      
      build_tree( v1 , a , med );
      build_tree( v2 , med + 1 , b );
      T[ node ] = pull( T[ v1 ] , T[ v2 ] );
}
void get_arrays( int node , int a , int b , int x , int y )
{
      int v1 = 2*node + 1 , v2 = 2*node + 2 , med = ( a + b )/2;      
      if( x > b || a > y ) return;
      if( a >= x && y >= b )
      {
            v.push_back( node );
            return;      
      }
      get_arrays( v1 , a , med , x , y );
      get_arrays( v2 , med + 1 , b , x , y );
}
int f( int t )
{
      int cnt = 0 ;
      for( int u = 0 ; u < v.size(); ++u )
            cnt += upper_bound( T[v[u]].begin() , T[v[u]].end() , A[ t ] ) - T[v[u]].begin();      
      return cnt;
}
int query( int x , int y , int z )
{
      v.clear();
      get_arrays( 0 , 0 , n - 1 , x , y );
      //sort( v.begin() , v.end() );
      int lo = 0 , hi = n - 1;
      if( f(lo) >= z )
            return A[lo];
      while( hi - lo > 1 )
      {
            int med = ( lo + hi )/2;
            if( f( med ) >= z )
                  hi = med;
            else  lo = med;
      }
      return A[hi];
}
int main()
{
      scanf( "%d%d" , &n , &Q );
      for( int i = 0 ; i < n ; ++i )
            scanf( "%d" , &A[i] );
      build_tree( 0 , 0 , n - 1 );
      sort( A , A + n );
      while( Q-- )
      {
            int i , j , k;
            scanf( "%d%d%d" , &i  , &j  , &k );
            i--,j--;
            printf( "%d\n" , query( i ,  j , k ) );
      }
}
########################## SPARSE TABLE (RMQ) #################################
// SPOJ 11772. Negative Score
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define LOGN 18
#define N 100000

typedef vector< int > vi;

int rmq[ LOGN + 1 ][ N + 5 ];

int query( int a , int b ){ 
	int r = 31 - __builtin_clz( b - a + 1 );
	return min( rmq[ r ][ a ] , rmq[ r ][ b - (1<<r) + 1 ] );
}
int main(){
	int cases , x , n , Q , a , b;
	sc( cases );
	REP( tc , cases ){
		sc( n ) , sc( Q );
		REP( i , n ){
			sc( x );
			rmq[ 0 ][ i ] = x;
		}
		for( int k = 1 ; k <= LOGN ; ++k )
			for( int i = 0 ; i + (1<<k) <= n ; ++i )
				rmq[ k ][ i ] = min( rmq[ k - 1 ][ i ] , rmq[ k - 1 ][ i + (1<<(k - 1)) ] );
		printf( "Scenario #%d:\n" , tc + 1 );
		REP( i , Q ){
			sc( a ) , sc( b );
			a -- , b --;
			printf( "%d\n" , query( a , b ) );
		}
	}
}
///////Sliding RMQ
struct slidingRmq{//maxi
	deque< pii > vec;
	slidingRmq(){}
	void clear(){
		vec.clear();
	}
	void insert( int x , int pos ){
		while( !vec.empty() && vec.back().fi < x ) vec.pop_back();
		vec.pb( mp( x , pos ) );
	}
	void erase( int x , int pos ){
		if( !vec.empty() && vec[ 0 ] == mp( x , pos ) ) vec.pop_front();
	}
	int getMaxi(){
		//assert( SZ(vec) );
		return vec[ 0 ].fi;
	}
}RMQ;
#-----------------------------------------------------------------#
########################## SQRT DECOMPOSITION #################################
//Codeforces Round #307 (Div. 2) E. GukiZ and GukiZiana
typedef long long ll;
typedef vector< ll > vll;
typedef pair< int , int > pii;
typedef vector< pii > vpii;
typedef vector< vpii > vvpii;
typedef vector< int > vi;
typedef vector< vi > vvi;
typedef vector< vvi > vvvi;
typedef vector< ll > vll;
typedef vector< vll > vvll;

int getLower( vi &v , int x ){
	return lower_bound( all( v ) , x ) - v.begin();
}
int getUpper( vi &v , int x ){
	return upper_bound( all( v ) , x ) - v.begin();
}

int getLower( vll &v , ll x ){
	return lower_bound( all( v ) , x ) - v.begin();
}
int getUpper( vll &v , ll x ){
	return upper_bound( all( v ) , x ) - v.begin();
}

int getLower( vpii &v , ll x ){
	pii par = mp( x , INT_MIN );
	return lower_bound( all( v ) , par ) - v.begin();
}

int getUpper( vpii &v , ll x ){
	pii par = mp( x , INT_MAX );
	return upper_bound( all( v ) , par  ) - v.begin();
}


void fix( int buck , vvll &buckets , vll &flag , vvpii &bucketsSorted , 
	int lo , int hi , int x , vvi &index ){
	
	for( auto &val : buckets[ buck ] ) val += flag[ buck ];
	flag[ buck ] = 0;

	for( int pos = lo ; pos <= hi ; ++pos )
		buckets[ buck ][ pos ] += x;
	vpii vec;
	REP( i , SZ( buckets[ buck ] ) )
		vec.pb( mp( buckets[ buck ][ i ] , index[ buck ][ i ] ) );
	sort( all( vec ) );
	bucketsSorted[ buck ] = vec;
}

int main(){
	int n , Q;
	while( sc( n ) == 1 ){
		sc( Q );
		vi A( n );
		int m = (int) sqrt( n );
		int len = (n - 1)/m + 1;
		vvll buckets( len );
		vvi index( len );
		vll flag( len );
		vi ind( n );
		vi lo( len , INT_MAX ) , hi( len , INT_MIN );
		REP( i , n ) sc( A[ i ] );

		vvpii bucketsSorted( len );		
		
		REP( i , n ){
			int bucket = i / m;
			buckets[ bucket ].pb( A[ i ] );
			lo[ bucket ] = min( lo[ bucket ] , i );
			hi[ bucket ] = max( hi[ bucket ] , i );
			index[ bucket ].pb( i );
			ind[ i ] = bucket;
		}
		
		REP( buck , len ) fix( buck , buckets , flag , bucketsSorted , 0 , 0 , 0 , index );
	
		REP( q , Q ){
			int op;
			sc( op );
			if( op == 1 ){
				int L , R , x;
				scanf( "%d%d%d" , &L , &R , &x );
				L -- , R -- ;
				int inda = getLower( lo , L );
				int indb = getUpper( hi , R );
				indb --;
				if( ind[ L ] == ind[ R ] ){
					int buck = ind[ L ];
					int a = getLower( index[ buck ] , L );
					int b = getLower( index[ buck ] , R );
					fix( buck , buckets , flag , bucketsSorted , a , b , x , index );
					continue;
				}else{
					if( inda <= indb ){
						for( int buck = inda ; ; buck ++ ){
							flag[ buck ] += x;
							if( buck == indb ) break;
						}
					}
					if( lo[ inda ] != L ){
						int buck = inda - 1;
						int a = getLower( index[ buck ] , L );
						int b = SZ( buckets[ buck ] ) - 1;
						fix( buck , buckets , flag , bucketsSorted , a , b , x , index );
					}
					if( hi[ indb ] != R ){
						int buck = indb + 1;
						int a = 0;
						int b = getLower( index[ buck ] , R );
						fix( buck , buckets , flag , bucketsSorted , a , b , x , index );
					}
					
				}
			}else{
				int y;
				sc( y );
				int a = INT_MAX , b = INT_MIN;
				REP( buck , len ){
					ll cur = (ll)y - flag[ buck ];
					int posa = getLower( bucketsSorted[ buck ] , cur );
					int posb = getUpper( bucketsSorted[ buck ] , cur );
					if( posa == posb ) continue;
					posb --;
					a = min( a , (int)bucketsSorted[ buck ][ posa ].se );
					b = max( b , (int)bucketsSorted[ buck ][ posb ].se );
				}
				if( a == INT_MAX ){
					puts( "-1" );
					continue;
				}
				printf( "%d\n" , b - a );
			}
			
		}
	}
}

########################## MERGESORT #################################
//SPOJ 3370. Mergesort
#define N 100005
int n ;
int A[ N ] , aux[ N ];
void merge_sort( int x , int y ){
	if( x == y )return;
	int med = ( x + y )>>1 , i = x , j = med + 1 , pos = x;
	merge_sort( x , med ) , merge_sort( med + 1 , y );
	while( i <= med && j <= y )
		aux[ pos++ ] = ( A[ i ] < A[ j ] ? A[ i++ ] : A[ j++ ]);
	while( i <= med ) aux[ pos++ ] = A[ i++ ];
	while( j <= y ) aux[ pos++ ] = A[ j++ ];
	for( int k = x ; k <= y ; ++k )
		A[ k ] = aux[ k ];
}
########################## CountInversions ##################################
// Formally speaking, two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j
// CountInversions = minimo numero de swaps(solo intercambiado dos elementos vecinos) que se necesita para volver un arreglo ordenado.
// Version con Busqueda binaria nlog(n)^2
#define N 200005
int n ;
int A[ N ] , aux[ N ];
ll InvCount( int x , int y ){
	if( x == y )return 0;
	int med = ( x + y )>>1 , i = x , j = med + 1 , pos = x;
	ll ans = InvCount( x , med ) + InvCount( med + 1 , y );
	while( i <= med && j <= y )
		aux[ pos++ ] = ( A[ i ] < A[ j ] ? A[ i++ ] : A[ j++ ]);
	for( int k = x ; k <= med ; ++k )
		ans += upper_bound( A + med + 1 , A + y + 1 , A[ k ] ) - ( A + med + 1 );
	while( i <= med ) aux[ pos++ ] = A[ i++ ];
	while( j <= y ) aux[ pos++ ] = A[ j++ ];
	for( int k = x ; k <= y ; ++k )
		A[ k ] = aux[ k ];
	return ans;
}
//// Formally speaking, two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j
#define N 200005
int n ;
int A[ N ] , aux[ N ];
ll InvCount( int x , int y ){
	if( x == y )return 0;
	int med = ( x + y )>>1 , i = x , j = med + 1 , pos = x;
	ll ans = InvCount( x , med ) + InvCount( med + 1 , y );
	while( i <= med && j <= y )
		if( A[ i ] < A[ j ] ){
			aux[ pos++ ] = A[ i++ ];
			ans += j - (med + 1);
		}
		else aux[ pos++ ] = A[ j++ ];
	while( i <= med ) aux[ pos++ ] = A[ i++ ] , ans += j - (med + 1);
	while( j <= y ) aux[ pos++ ] = A[ j++ ];
	for( int k = x ; k <= y ; ++k )
		A[ k ] = aux[ k ];
	return ans;
}
//checar el histograma ( idea sonnycson )
#-----------------------------------------------------------------#
############################# Treap( Roy - Emaxx ) ###############################################
// Treap(Treap) ( Set Map ) not implicit
//SPOJ 3273. Order statistic set
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define INF (INT_MAX)

typedef pair< int , int > pii;
typedef long long ll;
typedef vector< int > vi;

// treap (simple) implementation stores unique elements in key
struct Node{
	int key , prior;
	int cnt;
	Node *L , *R;
	Node(){}
	Node( int key , int prior ) : key( key ) , prior( prior ) , cnt( 1 )  ,L( 0 ) , R( 0 ) {}
};
typedef Node * pNode;
//getters and setter
inline int cnt( pNode &it ){ return it ? it -> cnt : 0; }
void upd_cnt( pNode &it ){
	if( it ) it -> cnt = cnt( it -> L ) + cnt( it -> R ) + 1;
}
//End getters and setters


//Basic Functions

// remember L.key < key < R.key  (BST)
// split treap t in L and R when in L is stored all values of t <= key and in R the rest
void split( pNode t , int key , pNode &L , pNode &R ){
	if( !t ){
		L = R = 0;
		upd_cnt( t );
		return;
	}
	if( t -> key > key ) split( t -> L , key , L , t -> L ) , R = t;
	else split( t -> R , key , t -> R , R ) , L = t;
	upd_cnt( t );
}
// merge works asuming that all keys of subtree L is less to all keys of subtree R
void merge( pNode & t , pNode L , pNode R ){
	if( !L ) {
		t = R;
		upd_cnt( t );
		return;
	}
	if( !R ) {
		t = L;
		upd_cnt( t );
		return;
	}
	if( (L -> prior) > (R -> prior) ) merge( L -> R , L -> R , R ) , t = L;
	else merge( R -> L , L , R -> L ) , t = R;
	upd_cnt( t );
}
// End Basic Functions

//Find returns true if key is conteinend in T
bool Find( pNode &t , int key ){
	if( !t ) return 0;
	if( (t -> key) == key ) return 1;
	return Find( key > (t -> key) ? (t -> R) : (t -> L) , key );
}
//

//insert in treap
void insert( pNode &t , pNode it ){
	if( !t ){
		t = it;
		upd_cnt( t );
		return;
	}
	if( (it -> prior) > (t -> prior) ){
		split( t , it -> key , it -> L , it -> R ) , t = it;
	}else{
		insert( (t -> key) < (it -> key) ? (t -> R) : (t -> L) , it );
	}
	upd_cnt( t );
}
//safely insert in treap
void insert( pNode &t , int key ){
	if( !Find( t , key ) ) 
		insert( t , new Node( key , rand() ) );
}
//erase
void erase( pNode &t , int key ){
	if( !t ) return;
	if( (t -> key) == key ) {
		merge( t , t -> L , t -> R );
		upd_cnt( t );
		return;
	}
	erase( key < (t -> key) ? (t -> L) : (t -> R) , key );
	upd_cnt( t );
}

//count less than key
int countLess( pNode &t , int key ){
	if( !t ) return 0;
	if( (t -> key) == key ) return cnt( t -> L );
	if( (t -> key) > key ) return countLess( t -> L , key );
	return 1 + cnt( t -> L ) + countLess( t -> R , key );
}
int FindKth( pNode &t , int k ){
	int L = cnt( t -> L );
	if( L + 1 == k ) return t -> key;
	if( L >= k ) return FindKth( t -> L , k );
	return FindKth( t -> R , k - (1 + L) );
}
//cuando se crea de esta manera T esta en null :)
pNode T;
int main(){	
	int Q;
	sc( Q );
	char op[ 4 ];
	REP( i , Q ){
		int x;
		scanf( "%s%d" , op , &x );
		if( op[ 0 ] == 'I' ){
			insert( T , x );
		}else if( op[ 0 ] == 'D' ){
			erase( T , x );
		}else if( op[ 0 ] == 'K' ){
			if( cnt( T ) < x ) {
				puts( "invalid" );
			}else{
				printf( "%d\n", FindKth( T , x ) );
			}
		}else{

			printf( "%d\n" , countLess( T , x ) );
		}
	}
}

///GCD 2010 Timus
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

typedef pair< int , int > pii;
typedef long long ll;
typedef vector< int > vi;

struct Node{
	int key , prior;
	int cnt;
	int g;
	Node *L , *R;
	Node(){}
	Node( int key , int prior ) : key( key ) , prior( prior ) , cnt( 1 )  , L( 0 ) , R( 0 ) , g( key ) {}
};
typedef Node * pNode;
inline int cnt( pNode &t ){ return t ? t -> cnt : 0; }
inline int g( pNode &t ){ return t ? t -> g : 0; }
void upd( pNode &it ){
	if( !it ) return;
	it -> cnt = cnt( it -> L ) + cnt( it -> R ) + 1;
	it -> g = __gcd( it -> key  , __gcd( g( it -> L ) , g( it -> R ) ) );
}
void split( pNode t , int key , pNode &L , pNode &R ){
	if( !t ){
		L = R = 0;
		upd( t );
		return;
	}
	if( t -> key > key ) split( t -> L , key , L , t -> L ) , R = t;
	else split( t -> R , key , t -> R , R ) , L = t;
	upd( t );
}
void merge( pNode & t , pNode L , pNode R ){
	if( !L ) {
		t = R;
		upd( t );
		return;
	}
	if( !R ) {
		t = L;
		upd( t );
		return;
	}
	if( (L -> prior) > (R -> prior) ) merge( L -> R , L -> R , R ) , t = L;
	else merge( R -> L , L , R -> L ) , t = R;
	upd( t );
}

void insert( pNode &t , pNode it ){
	if( !t ){
		t = it;
		upd( t );
		return;
	}
	if( (it -> prior) > (t -> prior) )
		split( t , it -> key , it -> L , it -> R ) , t = it;
	else
		insert( (t -> key) < (it -> key) ? (t -> R) : (t -> L) , it );
	upd( t );
}
bool Find( pNode &t , int key ){
	if( !t ) return 0;
	if( (t -> key) == key ) return 1;
	return Find( key > (t -> key) ? (t -> R) : (t -> L) , key );
}
void insert( pNode &t , int key ){
	if( !Find( t , key ) ) 
		insert( t , new Node( key , rand() ) );
	upd( t );
}
void erase( pNode &t , int key ){
	if( !t ) return;
	if( (t -> key) == key ) {
		merge( t , t -> L , t -> R );
		upd( t );
		return;
	}
	erase( key < (t -> key) ? (t -> L) : (t -> R) , key );
	upd( t );
}
void output( pNode &t ){
	if( !t )  return;
	output( t -> L );
	cout << " " << (t -> key);
	output ( t -> R );
}
int getId( vi &v , int val ){ return lower_bound( all( v ) , val ) - v.begin();}

pNode T;
int main(){	
	ios_base :: sync_with_stdio( 0 );
	int Q;
	cin >> Q;
	vi vec;
	vector< char > op;
	vi values;
	REP( q , Q ){
		char c;
		int val;
		cin >> c >> val;
		op.pb( c );
		values.pb( val );
	}
	vec = values;
	sort( all( vec ) );
	vec.resize( unique( all( vec ) ) - vec.begin() );
	vi mapa( SZ( vec ) );
	REP( q , Q ){
		char c;
		int val;
		c = op[ q ] , val = values[ q ];
		int id = getId( vec , val );
		if( c == '+' ){
			mapa[ id ] ++;
			if( mapa[ id ] == 1 )
				insert( T , val );
		}else{
			mapa[ id ] --;
			if( !mapa[ id ] )
				erase( T , val );
		}
		cout << max( 1 , g( T ) ) << '\n';
	}
}
//Implicit treap
//Arab 2012 LA 6319 - No Name
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define INF (INT_MAX)

typedef pair< int , int > pii;
typedef long long ll;
typedef vector< int > vi;

//Implicit treap as treap(treap) we have key prior and cnt is the magical variable :)
//Implicit treap basicly represents a linear array of integers ans we can do funny operations as:
//	1-insert a element
//  2-erase a element
//  3-calculate a asociative function in all treap
//  4-calculate a asociative function in a arbitrary subsegment(based in op 3) (through spliting in some treaps , answering then merging this treaps "for mass conservation :P" )
// all of this in log( n ) time :D

struct Node{
	int key , prior;
	int cnt;
	Node *L , *R;
	Node(){}
	Node( int key , int prior ) : key( key ) , prior( prior ) , cnt( 1 ) , L( 0 ) , R( 0 ) {}
};
typedef Node * pNode;
inline int cnt( pNode &t ){ return it ? it -> cnt : 0; }
void upd_cnt( pNode &it ){
	if( it ) it -> cnt = cnt( it -> L ) + cnt( it -> R ) + 1;
}
//Let's talking about cnt a.k.a "magical variable" , cnt is simply the size of the subtree of treap
//and we have to forget the order of key
// remember in treap(treap) L.key < key < R.key  (BST)
// then we change this order for a new order , the position in the array(of treap) that is represented by cnt is the quantity of elements in the left subtree


//merge is the same as treap(treap)
void merge( pNode & t , pNode L , pNode R ){
	if( !L ) {
		t = R;
		upd_cnt( t );
		return;
	}
	if( !R ) {
		t = L;
		upd_cnt( t );
		return;
	}
	if( (L -> prior) > (R -> prior) ) merge( L -> R , L -> R , R ) , t = L;
	else merge( R -> L , L , R -> L ) , t = R;
	upd_cnt( t );
}
//split changes a bit because we have the magical variable "cnt" 
// split treap t in L and R when in L is stored all values of t <= key and in R the rest THEN this means that we have a segment(treap)
// that after split we cut this segment in two parts (two treaps) :D
// as example we have [ 0 , n - 1 ] segment and we spit in L all elements from [ 0 , pos - 1 ] and [ pos , n - 1 ] in L
//split divides [ 0 , n - 1 ] in two parts [ 0 , pos - 1 ] and [ pos , n - 1 ] 
void split( pNode t , pNode &L , pNode &R , int key ){
    if( !t ) {
		L = R = 0;
		return;
	}
    int cntL = cnt(t->L);
    if (key <= cntL)
        split( t -> L , L , t -> L , key ) , R = t;
    else
        split( t -> R , t -> R , R , key - (cntL + 1) ), L = t;
    upd_cnt( t );
}
void output( pNode &t ){
	if( !t )  return;
	output( t -> L );
	cout << char( (t -> key) );
	output ( t -> R );
}

void insert( pNode &t , int pos , int key ){
	pNode nt = new Node( key , rand() );
	split( t , nt -> L , nt -> R , pos );
	t = nt;
	upd( t );
}

void erase( pNode &t , int pos) {
    if( !t ) return;
    int cntL = cnt( t -> L );
    if( pos < cntL ) erase( t -> L , pos );
    else if( pos == cntL ) merge( t , t -> L , t -> R );
    else erase( t -> R , pos - (cntL + 1) );
    upd_cnt( t );
}
void init( pNode &T , string &s ){
	T = 0;
	REP( i , SZ( s ) ) insert( T , i , s[ i ] );
}
int main(){
	ios_base :: sync_with_stdio( 0 );
	pNode T;
	int cases;
	cin >> cases;
	REP( tc , cases ){
		string s;
		cin >> s;
		//clearing treap
		
		init( T , s );
		//cout << "S: " << s << endl;
		while( 1 ){
			string op;
			cin >> op;
			if( op == "END" ) break;
			if( op == "I" ){
				string cad;
				int X;
				cin >> cad >> X;
				pNode t = 0 , L = 0 , R = 0;
				init( t , cad );
				split( T , L , R , X );
				merge( T , L , t );				
				merge( T , T , R );
			}else{
				int lo , hi;
				cin >> lo >> hi;
				pNode A = 0 , B = 0 , C = 0;
				split( T , A , B , lo );
				split( B , B , C , hi - lo + 1 );
				
				output( B );
				cout << '\n';
				
				merge( T , A , B );
				merge( T , T , C );
			}
		}
		
	}
}
//implicit treap with reverse(lazy propagation) and asociative funtion
//3961_ACM Robotic sort
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define INF (1<<29)

typedef pair< int , int > pii;
typedef long long ll;
typedef vector< int > vi;
typedef vector< pii > vpii;

struct Node{
	int key , prior;
	int cnt;
	bool rev;
	int mini;
	Node *L , *R;
	Node(){}
	Node( int key , int prior ) : key( key ) , prior( prior ) , cnt( 1 ) , L( 0 ) , R( 0 ) , rev( 0 ) , mini( key ) {}
};
typedef Node * pNode;
inline int cnt( pNode t ){ return t ? t -> cnt : 0; }
inline int mini( pNode t ){ return t ? t -> mini : INF; }
void upd( pNode t ){
	if( !t ) return;
	t -> cnt = cnt( t -> L ) + cnt( t -> R ) + 1;
	t -> mini = min( t -> key , min( mini( t -> L ) , mini( t -> R ) ) );
}

void push( pNode t ){
	if( !t || !(t -> rev) ) return;
	t -> rev = 0;
	swap( (t -> L) , (t -> R) );
	if( (t -> L) ) t -> L -> rev ^= 1;
	if( (t -> R) ) t -> R -> rev ^= 1;
}
void merge( pNode &t , pNode L , pNode R ){
	push( L ) , push( R );
	if( !L ) {
		t = R;
		upd( t );
		return;
	}
	if( !R ) {
		t = L;
		upd( t );
		return;
	}
	if( (L -> prior) > (R -> prior) ) merge( L -> R , L -> R , R ) , t = L;
	else merge( R -> L , L , R -> L ) , t = R;
	upd( t );
}

// split  [ 0 , pos - 1 ] [ pos , n - 1 ]
void split( pNode t , pNode &L , pNode &R , int key ){
	push( t );
	if( !t ) {
		L = R = 0;
		return;
	}
    int cntL = cnt(t->L);
    if (key <= cntL)
		split( t -> L , L , t -> L , key ) , R = t;
    else
        split( t -> R , t -> R , R , key - (cntL + 1) ), L = t;
    upd( t );
}


void init( pNode &T , vi &v ){
	T = 0;
	REP( i , SZ( v ) ) merge( T , T , new Node( v[ i ] , rand() ) );
}
void reverse( pNode &t , int lo , int hi ){
	push( t );
	pNode t1 = 0 , t2 = 0 , t3 = 0;
	split( t , t1 , t2 , lo );
	split( t2 , t2 , t3 , hi - lo + 1 );
	t2 -> rev ^= 1;
	push( t2 );
	merge( t , t1 , t2 );
	merge( t , t , t3 );
	push( t );
}

int getMin( pNode &t ){
	push( t );
	int ans = 0;
	if( (t -> mini) == mini( t -> L ) ) ans = getMin( t -> L );
	else if( (t -> mini) == mini( t -> R ) ) ans = cnt( t -> L ) + 1 + getMin( t -> R );
	else{
		ans = cnt(t->L);
	}
    upd( t );
	return ans;
}
void dfs( pNode &t ){
	if( !t )return;
	dfs( t -> L );
	cout << t -> key << " ";
	dfs( t -> R );
}
int query( pNode &t , int lo , int hi ){
	push( t );
	pNode t1 = 0 , t2 = 0 , t3 = 0;
	split( t , t1 , t2 , lo );
	split( t2 , t2 , t3 , hi - lo + 1 );
	push( t2 );

	int ans = getMin( t2 );

	merge( t , t1 , t2 );
	merge( t , t , t3 );
	push( t );
	return ans;
}

int main(){
	srand( time( 0 ) );
	int n;
	while( sc( n ) == 1 ){
		if( !n ) break;
		pNode T = 0;
		vi v( n );
		vpii w;
		REP( i , n ) {	
			int x;
			sc( x );
			w.pb( mp( x , i ) );
		}
		sort( all( w ) );
		REP( i , n ) v[ w[ i ].se ] = i;
		init( T , v );

		vi vec;
		REP( i , n ){
			int p = query( T , i , n - 1 );
			vec.pb( i + p );
			reverse( T , i , i + p );
		}
		printf( "%d" , vec[ 0 ] + 1 );
		for( int i = 1 ; i < SZ( vec ) ; ++i ) printf( " %d" , vec[ i ] + 1 );
		puts( "" );
		T -> rev = 0;
	}
}

########################## BIT  #################################
//binary indexded tree
//INVCNT SPOJ
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define N 200000
#define MAXVAL 10000000

typedef vector< int > vi;
typedef long long ll;

int bit[ MAXVAL + 5 ];

void update( int pos , int val ){
	while( pos <= MAXVAL ){
		bit[ pos ] += val;
		pos += (pos & -pos); 
	}
}
int query( int pos ){
	int sum = 0;
	while( pos ){
		sum += bit[ pos ];
		pos -= ( pos & -pos );
	}
	return sum;
}

int A[ N + 5 ];
int main(){
	int cases , x , n ;
	sc( cases );
	REP( tc , cases ){
		sc( n );
		REP( i , n ){
			sc( x );
			A[ i ] = x;
		}
		clr( bit , 0 );
		ll ans = 0 ;
		for( int i = n - 1 ; i >= 0 ; --i ){
			ans += query( A[ i ] );
			update( A[ i ] , 1 );
		}
		cout << ans << '\n';
	}
}

// se debe indexar desde 1 .... query( x + 1 ) , update( x + 1 , val )
// query returna el acumulado hasta pos
// update suma val en el elemento A[ pos ]
// updating ranges
// A ahora guarda diferencias consecutivas es decir A[ i ] = a[ i + 1 ] - a[ i ]
// query retorna el elemento a[ pos ]
// para update un rango a , b , val - 1 based 
// update( a , val )
// update( b + 1 , -val )

//DQuery count quanty of diferent in a subarray (queries)
//GYM / 100496D Data minig
#define MAXVAL 200000

int bit[ MAXVAL + 5 ];
int last[ MAXVAL + 5 ];
void upd( int pos , int val ){
	while( pos <= MAXVAL ){
		bit[ pos ] += val;
		pos += (pos & -pos); 
	}
}
void update( int pos , int val ){
	upd( pos + 1 , val );
}
int qry( int pos ){
	int sum = 0;
	while( pos ){
		sum += bit[ pos ];
		pos -= ( pos & -pos );
	}
	return sum;
}
int Query( int pos ){ return qry( pos + 1 ); }

int getId( vi &v , int x ){
	return lower_bound( all( v ) , x ) - v.begin();
}
struct query{
	int lo , hi , id , ans;
	query(){}
	query( int lo , int hi , int id , int ans ) : lo( lo ) , hi( hi ) , id( id ) , ans( ans ){}
};
bool operator < ( const query &a , const query &b ){
	if( a.hi != b.hi ) return a.hi < b.hi;
	if( a.lo != b.lo ) return a.lo < b.lo;
	return a.id < b.id;
}

vi solve( vi &A , vector< query > &V ){
	vi ans( SZ( V ) );
	clr( bit , 0 );
	clr( last , -1 );
    int end = 0;
    for( auto q : V ){
    	int lo = q.lo , hi = q.hi , id = q.id;
    	if( lo == -1 ){//query calculada previamente
    		ans[ id ] = q.ans;
    		continue;
    	}
    	for( int j = end ; j <= hi ; ++j ){
               if( last[ A[ j ] ] == -1 ){
                    last[ A[ j ] ] = j;
                    update( j , 1 );
               }else{
					update( last[ A[ j ] ] , -1 );
                    update( j , 1 );                    
                    last[ A[ j ] ] = j;
               }
		}
		ans[ id ] = Query(hi) - Query(lo - 1);//number of diferents
		ans[ id ] ++;
		end = hi;
    }
	return ans;
}
int main(){
	freopen( "data.in" , "r" , stdin );
	freopen( "data.out" , "w" , stdout );
	int n;
	while( sc( n ) == 1 ){
		vi A( n );
		REP( i , n ) sc( A[ i ] );
		vi a = A;
		sort( all( a ) );
		a.resize( unique( all( a ) ) - a.begin() );
		vvi T( SZ( a ) );
		REP( i , n ) T[ getId( a , A[ i ] ) ].pb( i );

		int Q;
		sc( Q );
		vector< query > V;
		REP( i , Q ){
			int pos , K;
			sc( pos ) , sc( K );
			pos --;
			
			int ida = getId( a , A[ pos + K - 1 ] );
			int p = lower_bound( all( T[ ida ] ) , pos ) - T[ ida ].begin();
			K = T[ ida ][ p ];
			// [pos K - 1][k]
			if( pos == K ){
				V.pb( query( -1 , -1 , i , 1 ) );
				continue;
			}
			V.pb( query( pos , K - 1 , i , -1 ) );
		}
		REP( i , n ) A[ i ] = getId( a , A[ i ] );
		sort( all( V ) );
		vi ans = solve( A , V );
		for( auto x : ans ) printf( "%d\n" , x );
	}
}

##########################  BIT 2D #################################
int read( int idx , int idy )
{
   int sum = 0;
   while( idx > 0 )
   {
      int y1 = idy;
      while( y1 > 0 )
      {
         sum += tree[idx][y1];
         y1 -= (y1 & -y1);
      }
      idx -= (idx & -idx);
   }
   return sum;   
}
void update( int x , int y , int val )
{
   while( x <= MAXN )
   {
      int y1 = y;
      while( y1 <= MAXN )
      {
         tree[x][y1] += val;
         y1 += (y1 & -y1);   
      }
      x += (x&-x);
   }   
}
#-----------------------------------------------------------------#


######################################################################################
############################# GEOMETRIA COMPUTACIONAL ################################
######################################################################################
// Version Resumida

typedef long double ld;

const ld EPS = 1e-8;
const ld PI = acos(-1.0);

inline bool equals( ld x , ld y ){ return abs( x - y ) < EPS; }
inline bool Less( ld x , ld y ){ return x < y - EPS; }
inline bool LessOrEquals( ld x , ld y ){ return Less( x , y ) || equals( x , y ); }
struct Point{
	ld x , y;
	Point(){ x = y = 0; }
	Point( ld x , ld y ) : x( x ) , y( y ) {}
	ld dist(){ return hypot( x , y );}
	Point ort(){ return Point( -y , x ); }
	void print(){ printf( "punto : %.4f %.4f\n" , double(x) , double(y) );}
};
typedef Point Vector;
Point operator - ( const Point &A , const Point &B ){ return Point( A.x - B.x , A.y - B.y ); }
Point operator + ( const Point &A , const Point &B ){ return Point( A.x + B.x , A.y + B.y ); }
Point operator * ( const Point &A , const ld &k ){ return Point( A.x * k , A.y * k ); }
bool operator == ( const Point &A , const Point &B ){ return equals( A.x , B.x ) && equals( A.y , B.y ) ; }

ld dot( const Point &A , const Point &B ){ return A.x * B.x + A.y * B.y ; }
ld cross( const Point &A , const Point &B ){ return A.x * B.y - A.y * B.x; }
ld area( const Point &A , const Point &B , const Point &C ){ return cross( B - A , C - A ); }
ld dist( const Point &A , const Point &B ){ return (B - A).dist();}
bool operator <( const Point &A , const Point &B ){ return abs( A.x - B.x ) < EPS ? A.y < B.y : A.x < B.x ; }

Point lineIntersection( const Point &A , const Point &B , const Point &C , const Point &D ){
	return A + ( B - A )*( cross( C - A , D - C ) / cross( B - A , D - C ) );
}
bool between( const Point &A , const Point &B , const Point &P ){
	return 	min( A.x , B.x ) < P.x + EPS && P.x < max( A.x , B.x ) + EPS &&
			min( A.y , B.y ) < P.y + EPS && P.y < max( A.y , B.y ) + EPS;
}
bool onSegment( const Point &A , const Point &B , const Point &P ){
	return abs( area( A , B , P ) ) < EPS && between( A , B , P );
}
bool intersects( const Point &P1 , const Point &P2 , const Point &P3 , const Point &P4 ){
	ld A1 = area( P3 , P4 , P1 );
	ld A2 = area( P3 , P4 , P2 );
	ld A3 = area( P1 , P2 , P3 );
	ld A4 = area( P1 , P2 , P4 );
	if( ( ( A1 > EPS && A2 < -EPS ) || ( A1 < -EPS && A2 > EPS ) ) && 
		( ( A3 > EPS && A4 < -EPS ) || ( A3 < -EPS && A4 > EPS ) ) ) return 1;
	if( onSegment( P3 , P4 , P1 ) ) return 1;
	if( onSegment( P3 , P4 , P2 ) ) return 1;
	if( onSegment( P1 , P2 , P3 ) ) return 1;
	if( onSegment( P1 , P2 , P4 ) ) return 1;
	return 0;
}

typedef vector< Point > Polygon;
bool pointInPoly( const Polygon &Poly , const Point &A ){
    int nP = Poly.size() , cnt = 0;
    REP( i , nP ){
        int inf = i , sup = ( i + 1 )%nP;
        if( Poly[ inf ].y > Poly[ sup ].y ) swap( inf , sup );
        if( Poly[ inf ].y <= A.y && A.y < Poly[ sup ].y )
            if( area( A , Poly[ inf ] , Poly[ sup ] ) > EPS )
                cnt++;
    }
    return cnt&1;
}
// revisar caso cuando la entrada es un punto o puntos colineales
Polygon convexHull( Polygon P ){
	sort( all( P ) );
	int nP = P.size() , k = 0;
	Point H[ nP << 1 ];
	REP( i , nP ){
		while( k >= 2 && area( H[ k - 2 ] , H[ k - 1 ] , P[ i ] ) < EPS ) k--;
		H[ k++ ] = P[ i ];
	}
	for( int i = nP - 2 , sz = k; i >= 0 ; --i ){
		while( k > sz && area( H[ k - 2 ] , H[ k - 1 ] , P[ i ] ) < EPS ) k--;
		H[ k++ ] = P[ i ];
	}
	if( k == 0 ) return Polygon();
	return Polygon( H , H + k - 1 );
}

void AntipodalPairs( Polygon &P )
{
	int nP = P.size();
	for( int i = 0 , j = 2 ; i < nP ; ++i ){
		// P[j] debe ser el punto mas lejano a la linea P[i], P[(i+1)%n]:		
		while( 	area( P[ i ] , P[ ( i + 1 )%nP ] , P[ ( j + 1 )%nP ] ) - EPS > 
		     	area( P[ i ] , P[ ( i + 1 )%nP ] , P[ j ] ) ) j = ( j + 1 )%nP;

	}       
}
////
// line segment p-q intersect with line A-B.

// cuts polygon Q along the line formed by point a -> point b
// (note: the last point must be the same as the first point)
vector< Point > cutPolygon( Point a , Point b , Polygon Q ) {
	vector< Point > P;
	REP( i , SZ(Q) ){
		ld left1 = cross( b - a , Q[ i ] - a ) , left2 = 0;
		if( i != SZ(Q) - 1 ) left2 = cross( b - a , Q[ i + 1 ] - a );
		if( left1 > -EPS ) P.push_back( Q[ i ] );
		if (left1 * left2 < -EPS)
		P.push_back( lineIntersection( Q[ i ] , Q[ i + 1 ] , a , b ) );
	}
	
	if( !P.empty() && !(P.back() == P.front()) )
		P.push_back( P.front() );
	
	// make P’s first point = P’s last point
	return P;
}

/// Version con long double
/////// 12304_UVA - 2D Geometry 110 in 1!
typedef long double ld;

#define Vector Point
#define PI (acos( -1.0 ))
#define EPS (1e-8)

struct Point
{
	ld x , y;
	Point(){};
	Point( ld x , ld y ):x( x ) , y( y ){}
	Point ort(){ return Point( -y , x ); }
	ld arg(){ return atan2( y , x ); }
	ld get_angle(){ 
		ld t = arg();
		if( abs( t - PI ) < EPS )return 0;
		if( t < -EPS ) return t + PI ; 
		return t;
	}
	ld norm(){ return sqrt( x*x + y*y ); }
	Point unit(){ ld k = norm() ; return Point( x/k , y/k );}
	void read(){ cin >> x >> y; }
};
Point operator *( const Point &A , const ld &k ){ return Point( A.x*k , A.y*k );}
Point operator /( const Point &A , const ld &k ){ return Point( A.x/k , A.y/k );}
Point operator +( const Point &A , const Point &B ){ return Point( A.x + B.x , A.y + B.y );}
Point operator -( const Point &A , const Point &B ){ return Point( A.x - B.x , A.y - B.y );}
bool operator <( const Point &A , const Point &B ){ return mp( A.x , A.y ) < mp( B.x , B.y ); }
ld cross( const Point &A , const Point &B ){ return A.x*B.y - A.y*B.x; }
ld dot( const Point &A , const Point &B ){ return A.x*B.x + A.y*B.y; }
ld area( const Point &A , const Point &B , const Point &C ){ return cross( B - A , C - A );}
ld dist( const Point &A , const Point &B ){ return sqrt( dot( B - A , B - A ) ); }
bool isParallel(const Point &P1, const Point &P2, const Point &P3, const Point &P4){
	return abs( cross(P2 - P1, P4 - P3) ) <= EPS;
}
Point lineIntersection( const Point &A , const Point &B , const Point &C , const Point &D ){
	return A + ( B - A )*( cross( C - A , D - C ) / cross( B - A , D - C ) );
}
Vector Bis( const Point &A , const Point &B , const Point &C ){
	Vector V = ( B - A ) , W = ( C - A );
	Vector uV = V.unit() , uW = W.unit();
	return uV + uW;
}
ld distPointLine( const Point &A , const Point &B , const Point &P ){
	ld S = abs( area( A , B , P ) );
	ld L = dist( A , B );
	return S / L;
}

void CircumscribedCircle()
{
	Point X1 , X2 , X3;
	X1.read() , X2.read() , X3.read();
	Point N = ( X1 + X2 )/2.0 , M = ( X1 + X3 )/2.0;
	Point V = ( X2 - X1 ) , W = ( X3 - X1 );
	Point Inter = lineIntersection( N , N + V.ort() , M , M + W.ort() );
	printf( "(%.6f,%.6f,%.6f)\n" , double( Inter.x ) , double( Inter.y ) , double( dist( Inter , X1 ) ) );
}
void InscribedCircle()
{
	Point X1 , X2 , X3;
	X1.read() , X2.read() , X3.read();
	Vector V = Bis( X1 , X2 , X3 ) , W = Bis( X2 , X1 , X3 );
	Point Inter = lineIntersection( X1 , X1 + V , X2 , X2 + W );
	printf( "(%.6f,%.6f,%.6f)\n" , double( Inter.x ) , double( Inter.y ) , double( distPointLine( X1 , X2 , Inter ) ) );
}
void TangentLineThroughPoint()
{
	Point P , C;
	ld R;
	C.read() , cin >> R , P.read();
	Vector PC = C - P;
	ld dPC2 = dot( PC , PC );
	if( dPC2 >= R*R + EPS )
	{
		ld L = sqrt( dPC2 - R*R );
		ld ux , uy;
		ux = cross( PC , Point( -R , L ) )/dPC2;
		uy = cross( Point( L , R ) , PC )/dPC2;
		ld ang1 = Point( ux , uy ).get_angle()/PI*180.0;
		ux = cross( PC , Point( R , L ) )/dPC2;
		uy = cross( Point( L , -R ) , PC )/dPC2;
		ld ang2 = Point( ux , uy ).get_angle()/PI*180.0;
		if( ang1 > ang2 )swap( ang1 , ang2 );
		printf( "[%.6f,%.6f]\n" , double( ang1 ) , double( ang2 ) );
	}
	else if( abs( dPC2 - R*R ) <= EPS )
	{
		ld ang = PC.ort().get_angle()/PI*180.0;
		printf( "[%.6f]\n" , double( ang ) );
	}
	else puts( "[]" );
}
void CircleThroughAPointAndTangentToALineWithRadius()
{
	Point P , X1 , X2 ;
	ld R;
	P.read() , X1.read() , X2.read() , cin >> R;
	Vector w = ( X2 - X1 ).unit();
	Vector v = w.ort();
	if( distPointLine( X1 , X2 , P ) > 2*R + EPS ) puts( "[]" );
	else if( abs( area( X1 , X2 , P ) ) < EPS )
	{
		Point P1 = P - v*R , P2 = P + v*R;
		if( P2 < P1 )swap( P1 , P2 );
			printf( "[(%.6f,%.6f),(%.6f,%.6f)]\n" , double( P1.x ) , double( P1.y ) , double( P2.x ) , double( P2.y ) );
	}
	else
	{
		Point Q;
		if( area( X1 , X2 , P ) > 0 )
			Q = X1 + v*R;
		else Q = X1 - v*R;
		Point Inter = lineIntersection( Q , Q + w , P , P + v );
		ld H = dist( Inter , P );
		ld L = sqrt( R*R - H*H );
		if( L < EPS ) printf( "[(%.6f,%.6f)]\n" , double( Inter.x ) , double( Inter.y ) );
		else{
			Point P1 = Inter - w*L , P2 = Inter + w*L;
			if( P2 < P1 )swap( P1 , P2 );
			printf( "[(%.6f,%.6f),(%.6f,%.6f)]\n" , double( P1.x ) , double( P1.y ) , double( P2.x ) , double( P2.y ) );
		}
	}
}
void CircleTangentToTwoLinesWithRadius()
{
	Point X1 , X2 , X3 , X4;
	ld R;
	X1.read() , X2.read() , X3.read() , X4.read();
	cin >> R;
	Point Inter = lineIntersection( X1 , X2 , X3 , X4 );
	Vector v = ( X2 - X1 ).unit() , w = ( X4 - X3 ).unit();
	Vector u  = ( v + w ).unit();
	ld sent = abs( cross( v , u ) );
	Point P1 = Inter + u*(R/sent);
	Point P2 = Inter - u*(R/sent);
	u = u.ort();
	ld senp = sqrt( 1 - sent*sent );
	Point P3 = Inter + u*(R/senp);
	Point P4 = Inter - u*(R/senp);
	Point A[4] = { P1 , P2 , P3 , P4 };
	sort( A , A + 4 );
	REP( i , 4 )
		printf( "%s(%.6f,%.6f)%s" , (i==0?"[":"") , double( A[i].x ), double( A[i].y ) , (i+1==4?"]\n":",") );
}
void CircleTangentToTwoDisjointCirclesWithRadius()
{
	Point A , B;
	ld R1 , R2 , R;
	A.read() , cin >> R1;
	B.read() , cin >> R2;
	cin >> R;
	ld dAB = dist( A , B );
	if( 2*R < dAB - R1 - R2 - EPS )puts( "[]" );
	else
	{
		ld a = R + R1 , c = R + R2 , b = dAB;
		ld u = ( a*a + b*b - c*c )/(2*b);
		ld H = sqrt( a*a - u*u );
		Vector V = ( B - A ).unit();
		Vector W = V.ort();
		Point P = A + (V*u) + (W*H);
		Point Q = A + (V*u) - (W*H);
		if( Q < P )swap( P , Q );
		if( H < EPS )printf( "[(%.6f,%.6f)]\n" , double( P.x ) , double( P.y ) );
		else printf( "[(%.6f,%.6f),(%.6f,%.6f)]\n" , double( P.x ) , double( P.y ) ,double( Q.x ) , double( Q.y ) );
	}
}
ld area( vector<Point> &Poly )
{
	int nP = Poly.size();
	ld ans = 0;
	REP( i , nP )ans += area( O , Poly[i] , Poly[(i+1)%nP] );
	return abs( ans );
}
bool pointInPoly( vector< Point >&Poly , Point P ){
	int nP = Poly.size();
	int cnt = 0;
	Vector V( MOD1 , MOD2 );
	REP( i , nP )
	{
		if( onSegment( Poly[i] , Poly[(i+1)%nP] , P ) )return 0;
		cnt += intersects( Poly[i] , Poly[(i+1)%nP] , P , P + V );
	}
	return (cnt&1);
}
//////////////////// Representacion Implicita y Vectorial de Lineas ///////////////

struct Point
{
	double x , y;
	Point(){}
	Point( double x , double y ):x( x ) , y( y ){}
	Point ort(){ return Point( -y , x );}
};
Point O( 0 , 0 );
Point operator -( const Point &A , const Point &B ){ return Point( A.x - B.x , A.y - B.y );}
Point operator +( const Point &A , const Point &B ){ return Point( A.x + B.x , A.y + B.y );}
double dot( const Point &A ,const Point &B ){ return A.x*B.x + A.y*B.y;}
double cross( const Point &A , const Point &B ){ return A.x*B.y - A.y*B.x;}
Point operator*( const Point &A , const double k ){ return Point( A.x*k , A.y*k );}
bool isParallel(const Point &P1, const Point &P2, const Point &P3, const Point &P4)
{
	return abs( cross(P2 - P1, P4 - P3) ) <= EPS;
}
Point lineIntersection(const Point &A, const Point &B, const Point &C, const Point &D)
{
    return A + (B - A) * (cross(C - A, D - C) / cross(B - A, D - C));
}
struct Line
{
	int a , b , c;
	Line(){}
	Line( int a , int b , int c ):a(a),b(b),c(c){}
	void read(){ scanf( "%d%d%d", &a , &b , &c ); }
};
Line transf( const Point &P , const Point &Q )
{
	return Line( Q.y - P.y , P.x - Q.x , cross( P , Q ) );
}
//// Punto de paso y vector direccion

pair< Point , Point > transf( const Line &L )
{
	Point PO = Point( L.a , L.b );
	if( L.c == 0 )return mp( O , PO.ort() ) ;
	double cte = L.c/dot( PO , PO );
	return mp( PO*cte , PO.ort()*cte ) ;
}
///////////////////////////////////////////////////////////////////////////////////////////
/////// Version de Roy
#define EPS 1e-8
#define PI acos(-1)
#define Vector Point

struct Point
{
    double x, y;
    Point(){}
    Point(double a, double b) { x = a; y = b; }
    double mod2() { return x*x + y*y; }
    double mod()  { return sqrt(x*x + y*y); }
    double arg()  { return atan2(y, x); }
    Point ort()   { return Point(-y, x); }
    Point unit()  { double k = mod(); return Point(x/k, y/k); }
};
Point operator +(const Point &a, const Point &b) { return Point(a.x + b.x, a.y + b.y); }
Point operator -(const Point &a, const Point &b) { return Point(a.x - b.x, a.y - b.y); }
Point operator /(const Point &a, double k) { return Point(a.x/k, a.y/k); }
Point operator *(const Point &a, double k) { return Point(a.x*k, a.y*k); }
bool operator ==(const Point &a, const Point &b){ return fabs(a.x - b.x) < EPS && fabs(a.y - b.y) < EPS;}
bool operator !=(const Point &a, const Point &b){ return !(a==b);}
bool operator <(const Point &a, const Point &b){ if(a.x != b.x) return a.x < b.x; return a.y < b.y;}
double dist(const Point &A, const Point &B)    { return hypot(A.x - B.x, A.y - B.y); }
double cross(const Vector &A, const Vector &B) { return A.x * B.y - A.y * B.x; }
double dot(const Vector &A, const Vector &B)   { return A.x * B.x + A.y * B.y; }
double area(const Point &A, const Point &B, const Point &C) { return cross(B - A, C - A); }

double get_angle( Point A , Point P , Point B )
{
	double ang = (A-P).arg() - (B-P).arg(); 
	while(ang < 0) ang += 2*PI; 
	while(ang > 2*PI) ang -= 2*PI;
	return min(ang, 2*PI-ang);
}
bool isInt(double k){	return abs(k - int(k + 0.5)) < 1e-5;}
/////
//responde si un punto esta en un poligono convexo incluyendo la frontera
bool isIn( Point O , Point A , Point B , Point X )
{
	return abs( abs( area(O,A,B)) -( abs(area(O,A,X)) + abs(area(A,B,X)) + abs(area(O,B,X)) ) ) < EPS ;
}
// transforma el angulo de atan2 al rango 0 - 2PI
double f( double a )
{
	if( a < 0 )a += 2*PI;
	return a;
}
////////////////////////////////////////////////////////////////////////////////
// Heron triangulo y cuadrilatero ciclico
//http://en.wikipedia.org/wiki/Brahmagupta's_formula ----- sqrt((p-a) * (p-b) * (p-c)*(p-d))
// http://mathworld.wolfram.com/CyclicQuadrilateral.html
// http://www.spoj.pl/problems/QUADAREA/

//adicional existencia de trapezoide y altura
//http://en.wikipedia.org/wiki/Trapezoid

double areaHeron(double a, double b, double c)
{
	double s = (a + b + c) / 2;
	return sqrt(s * (s-a) * (s-b) * (s-c));
}

double circumradius(double a, double b, double c) { return a * b * c / (4 * areaHeron(a, b, c)); }

double areaHeron(double a, double b, double c, double d)
{
	double s = (a + b + c + d) / 2;
	return sqrt((s-a) * (s-b) * (s-c) * (s-d));
}

double circumradius(double a, double b, double c, double d) { return sqrt((a*b + c*d) * (a*c + b*d) * (a*d + b*c))  / (4 * areaHeron(a, b, c, d)); }

//### DETERMINA SI P PERTENECE AL SEGMENTO AB ###########################################
bool between(const Point &A, const Point &B, const Point &P)
{
    return  P.x + EPS >= min(A.x, B.x) && P.x <= max(A.x, B.x) + EPS &&
            P.y + EPS >= min(A.y, B.y) && P.y <= max(A.y, B.y) + EPS;
}

bool onSegment(const Point &A, const Point &B, const Point &P)
{
    return abs(area(A, B, P)) < EPS && between(A, B, P);
}

//### DETERMINA SI EL SEGMENTO P1Q1 SE INTERSECTA CON EL SEGMENTO P2Q2 #####################
//11343_UVA
bool intersects(const Point &P1, const Point &P2, const Point &P3, const Point &P4)
{
    double A1 = area(P3, P4, P1);
    double A2 = area(P3, P4, P2);
    double A3 = area(P1, P2, P3);
    double A4 = area(P1, P2, P4);
    
    if( ((A1 > 0 && A2 < 0) || (A1 < 0 && A2 > 0)) && 
        ((A3 > 0 && A4 < 0) || (A3 < 0 && A4 > 0))) 
            return true;
    
    else if(A1 == 0 && onSegment(P3, P4, P1)) return true;
    else if(A2 == 0 && onSegment(P3, P4, P2)) return true;
    else if(A3 == 0 && onSegment(P1, P2, P3)) return true;
    else if(A4 == 0 && onSegment(P1, P2, P4)) return true;
    else return false;
}

//### DETERMINA SI A, B, M, N PERTENECEN A LA MISMA RECTA ##############################
bool sameLine(Point P1, Point P2, Point P3, Point P4)
{
	return area(P1, P2, P3) == 0 && area(P1, P2, P4) == 0;
}
//### SI DOS SEGMENTOS O RECTAS SON PARALELOS ###################################################
bool isParallel(const Point &P1, const Point &P2, const Point &P3, const Point &P4)
{
	return abs( cross(P2 - P1, P4 - P3) ) <= EPS;
}

//### PUNTO DE INTERSECCION DE DOS RECTAS NO PARALELAS #################################
Point lineIntersection(const Point &A, const Point &B, const Point &C, const Point &D)
{
    return A + (B - A) * (cross(C - A, D - C) / cross(B - A, D - C));
}

Point circumcenter(const Point &A, const Point &B, const Point &C)
{
	return (A + B + (A - B).ort() * dot(C - B, A - C) / cross(A - B, A - C)) / 2;
}

//### FUNCIONES BASICAS DE POLIGONOS ################################################
bool isConvex(const vector <Point> &P)
{
    int nP = P.size(), pos = 0, neg = 0;
    for(int i=0; i<nP; i++)
    {
        double A = area(P[i], P[(i+1)%nP], P[(i+2)%nP]);
        if(A < 0) neg++;
        else if(A > 0) pos++;
    }
    return neg == 0 || pos == 0;
}

double area(const vector <Point> &P)
{
    int nP = P.size();
    double A = 0;
    for(int i=1; i<=nP-2; i++)
        A += area(P[0], P[i], P[i+1]);
    return abs(A/2);
}


//### DETERMINA SI A ESTA EN EL INTERIOR DEL POLIGONO( sin boundary ) ########################
// works in simple poly
bool pointInPoly(const vector <Point> &P, const Point &A)
{
    int nP = P.size(), cnt = 0;
    for( int i = 0 ; i < nP ; i++ )
    {
        int inf = i , sup = ( i + 1 )%nP;
        if( P[ inf ].y > P[ sup ].y ) swap( inf , sup );
        if( P[ inf ].y <= A.y && A.y < P[ sup ].y )
            if( area( A , P[ inf ] , P[ sup ] ) > 0 )
                cnt++;
    }
    return (cnt % 2) == 1;
}

/* TEOREMA DE PICK 
A = I + B/2 - 1, donde:


A = Area de un poligono de coordenadas enteras
I = Numero de puntos enteros en su interior
B = Numero de puntos enteros sobre sus bordes


Haciendo un cambio en la formula : I=(2A-B+2)/2, tenemos una forma de calcular
el numero de puntos enteros en el interior del poligono

int IntegerPointsOnSegment(const point &P1, const point &P2){

    point P = P1-P2;
    P.x = abs(P.x); P.y = abs(P.y);
    
    if(P.x == 0) return P.y;

    if(P.y == 0) return P.x;

    return (__gcd(P.x,P.y));
}



Se asume que los vertices tienen coordenadas enteras. Sumar el valor de esta
funcion para todas las aristas para obtener el numero total de punto en el borde

del poligono.

*/

// O(n log n)
// Entender que el convexhull te elimina los puntos repetidos :)

vector <Point> ConvexHull(vector <Point> Poly)
{
    sort(Poly.begin(),Poly.end());
    int nP = Poly.size(),k = 0;
    Point H[ 2*nP ];
    
    for( int i = 0 ; i < nP ; ++i ){
        while( k >= 2 && area( H [ k - 2 ] , H[ k - 1 ] , Poly[ i ] ) <= 0) --k;
        H[ k++ ] = Poly[ i ];
    }
    for( int i = nP - 2 , t = k ; i >= 0 ; --i ){
        while( k > t && area( H[ k - 2 ] , H[ k - 1 ] , Poly[ i ] ) <= 0) --k;
        H[ k++ ] = Poly[ i ];
    }
    if( k == 0 )return vector <Point>();
    return vector <Point> ( H , H + k - 1 );
}
//### DETERMINA SI P ESTA EN EL INTERIOR DEL POLIGONO CONVEXO A ########################

// O (log n)
bool isInConvex(vector <Point> &A, const Point &P)
{
	int n = A.size(), lo = 1, hi = A.size() - 1;
	
	if(area(A[0], A[1], P) <= 0) return 0;
	if(area(A[n-1], A[0], P) <= 0) return 0;
	
	while(hi - lo > 1)
	{
		int mid = (lo + hi) / 2;
		
		if(area(A[0], A[mid], P) > 0) lo = mid;
		else hi = mid;
	}
	
	return area(A[lo], A[hi], P) > 0;
}

// O(n)
Point norm(const Point &A, const Point &O)
{
    Vector V = A - O;
    V = V * 10000000000.0 / V.mod();
    return O + V;
}

bool isInConvex(vector <Point> &A, vector <Point> &B)
{
    if(!isInConvex(A, B[0])) return 0;
    else
    {
        int n = A.size(), p = 0;
        
        for(int i=1; i<B.size(); i++)
        {
            while(!intersects(A[p], A[(p+1)%n], norm(B[i], B[0]), B[0])) p = (p+1)%n;
            
            if(area(A[p], A[(p+1)%n], B[i]) <= 0) return 0;
        }
        
        return 1;
    }
}
       
//##### SMALLEST ENCLOSING CIRCLE O(n) ###############################################
// http://www.cs.uu.nl/docs/vakken/ga/slides4b.pdf
// http://www.spoj.pl/problems/ALIENS/

pair <Point, double> enclosingCircle(vector <Point> P)
{
    random_shuffle(P.begin(), P.end());
    
    Point O(0, 0);
    double R2 = 0;
    
    for(int i=0; i<P.size(); i++)
    {
        if((P[i] - O).mod2() > R2 + EPS)
        {
            O = P[i], R2 = 0;
            for(int j=0; j<i; j++)
            {
                if((P[j] - O).mod2() > R2 + EPS)
                {
                    O = (P[i] + P[j])/2, R2 = (P[i] - P[j]).mod2() / 4;
                    for(int k=0; k<j; k++)
                        if((P[k] - O).mod2() > R2 + EPS)
                            O = circumcenter(P[i], P[j], P[k]), R2 = (P[k] - O).mod2();
                }
            }
        }
    }
    return make_pair(O, sqrt(R2));
}

//##### CLOSEST PAIR OF POINTS ########################################################
bool XYorder(Point P1, Point P2)
{
	if(P1.x != P2.x) return P1.x < P2.x;
	return P1.y < P2.y;
}
bool YXorder(Point P1, Point P2)
{
	if(P1.y != P2.y) return P1.y < P2.y;
	return P1.x < P2.x;
}
double closest_recursive(vector <Point> vx, vector <Point> vy)
{
	if(vx.size()==1) return 1e20;
	if(vx.size()==2) return dist(vx[0], vx[1]);
	
	Point cut = vx[vx.size()/2];
	
	vector <Point> vxL, vxR;
	for(int i=0; i<vx.size(); i++)
		if(vx[i].x < cut.x || (vx[i].x == cut.x && vx[i].y <= cut.y))
			vxL.push_back(vx[i]);
		else vxR.push_back(vx[i]);
	
	vector <Point> vyL, vyR;
	for(int i=0; i<vy.size(); i++)
		if(vy[i].x < cut.x || (vy[i].x == cut.x && vy[i].y <= cut.y))
			vyL.push_back(vy[i]);
		else vyR.push_back(vy[i]);
	
	double dL = closest_recursive(vxL, vyL);
	double dR = closest_recursive(vxR, vyR);
	double d = min(dL, dR);
	
	vector <Point> b;
	for(int i=0; i<vy.size(); i++)
		if(abs(vy[i].x - cut.x) <= d)
			b.push_back(vy[i]);
	
	for(int i=0; i<b.size(); i++)
		for(int j=i+1; j<b.size() && (b[j].y - b[i].y) <= d; j++)
			d = min(d, dist(b[i], b[j]));
	
	return d;
}
double closest(vector <Point> points)
{
	vector <Point> vx = points, vy = points;
	sort(vx.begin(), vx.end(), XYorder);
	sort(vy.begin(), vy.end(), YXorder);
	
	for(int i=0; i+1<vx.size(); i++)
		if(vx[i] == vx[i+1])
			return 0.0;
	
	return closest_recursive(vx,vy);
}
////Codeforces Round #245 (Div. 1) D. Tricky Function 
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) (int) v.size()

#define INF 1e100
#define EPS (1e-8)
#define N 100000

typedef long long ll;
typedef double ld;
typedef vector< int > vi;

inline bool equals( ld x , ld y ){ return abs( x - y ) < EPS; }
inline bool Less( ld x , ld y ){ return x < y - EPS; }
inline bool LessOrEquals( ld x , ld y ){ return Less( x , y ) || equals( x , y ); }
struct Point{
    ld x , y;
    Point(){ x = y = 0; }
    Point( ld x , ld y ) : x( x ) , y( y ) {}
    ld dist(){ return sqrt( (ld) x * x + (ld)y * y ); }
};
Point operator -( const Point &A , const Point &B ){ return Point( A.x - B.x , A.y - B.y );}
bool operator ==(const Point &A, const Point &B){ return equals( A.x , B.x ) && equals( A.y , B.y );}
ld dist( const Point &A , const Point &B ){ return (B - A).dist();}
bool XYorder( Point P1 , Point P2 ){
    if( !equals( P1.x , P2.x ) ) return Less( P1.x , P2.x );
    return Less( P1.y , P2.y );
}
bool YXorder(Point P1, Point P2){
	if( !equals( P1.y , P2.y ) ) return Less( P1.y , P2.y );
	return Less( P1.x , P2.x );
}
double closest_recursive( vector < Point > vx , vector < Point > vy ){
    if( SZ( vx ) == 1 ) return 1e20;
    if( SZ( vx ) == 2 ) return dist( vx[ 0 ] , vx[ 1  ] );
    Point cut = vx[ SZ( vx ) >> 1 ];
    
    vector <Point> vxL, vxR;
    REP( i , SZ( vx ) )
        if( Less( vx[i].x , cut.x ) || ( equals( vx[i].x , cut.x ) && LessOrEquals( vx[i].y , cut.y ) ) )
            vxL.pb( vx[ i ] );
        else vxR.pb( vx[ i ] );
    
    vector < Point > vyL, vyR;
    REP( i , SZ( vy ) )
    	if( Less( vy[i].x , cut.x ) || ( equals( vy[i].x , cut.x ) && LessOrEquals( vy[i].y , cut.y ) ) )
            vyL.pb( vy[ i ] );
        else vyR.pb( vy[ i ] );
    
    double dL = closest_recursive( vxL , vyL );
    double dR = closest_recursive( vxR , vyR );
    double d = min( dL , dR );
    
    vector <Point> b;
    REP( i , SZ( vy ) )
        if( abs( vy[ i ].x - cut.x ) <= d + EPS )
            b.push_back( vy[ i ] );
    
    REP( i , SZ( b ) )
        for(int j = i + 1 ; j < SZ( b ) && (b[ j ].y - b[ i ].y) <= d + EPS ; j++)
            d = min( d , dist( b[ i ] , b[ j ] ) );
    
    return d;
}
double closest(vector <Point> points )
{
    vector <Point> vx = points, vy = points;
    sort( all( vx ) , XYorder );
    sort( all( vy ) , YXorder);
    REP( i , SZ( vx ) - 1 )
        if( vx[ i ] == vx[ i + 1 ] )
            return 0.0;
    return closest_recursive( vx , vy );
}
// CLOSEST WITH SWEEP LINE

#define INF 1e100
#define EPS (1e-8)

struct Point{
	ld x , y;
	Point(){ x = y = 0; }
	Point( ld x , ld y ) : x( x ) , y( y ) {}
	ld dist(){ return sqrt( (ld) x * x + (ld)y * y ); }
};
Point operator -( const Point &A , const Point &B ){ return Point( A.x - B.x , A.y - B.y );}
ld dist( const Point &A , const Point &B ){ return (B - A).dist();}
struct cmpX{
	bool operator()( const Point &A , const Point &B ) const{ 
		if( abs( A.x - B.x ) >= EPS ) return A.x < B.x - EPS; 
		return A.y < B.y - EPS;
	}
};
struct cmpY{
	bool operator()( const Point &A , const Point &B ) const{
		if( abs( A.y - B.y ) >= EPS ) return A.y < B.y - EPS;
		return A.x < B.x - EPS;
	}
};

int main()
{
	int n ;
	ld x , y ;
	while( cin >> n ){
		if( !n ) break;
		vector< Point > V( n );
		REP( i , n ){
			cin >> x >> y;
			V[ i ] = Point( x , y );
		}
		sort( all( V ) , cmpX() );
		set< Point , cmpY > SET;
		set< Point , cmpY >::iterator y1 , y2 , y3 ;
		int lo = 0 ;
		ld H = INF;
		REP( i , n ){
			while( H < V[ i ].x - V[ lo ].x - EPS ) 
				SET.erase( V[ lo ++ ] );
			y1 = lower_bound( all( SET ) , Point( -INF , V[ i ].y - H ) , cmpY() );
			y2 = upper_bound( all( SET ) , Point( INF , V[ i ].y + H ) , cmpY() );
			for( y3 = y1 ; y3 != y2 ; ++y3 )
				H = min( H , dist( V[ i ] , *y3 ) );
			SET.insert( V[ i ] );
		}
		printf( "%.4lf\n" , double( H ) );
	}
}

// INTERSECCION DE CIRCULOS
vector <Point> circleCircleIntersection(Point O1, double r1, Point O2, double r2)
{
	vector <Point> X;
	double d = dist(O1, O2);
	if(d > r1 + r2 || d < max(r2, r1) - min(r2, r1)) return X;
	else
	{
		double a = (r1*r1 - r2*r2 + d*d) / (2.0*d);
		double b = d - a;
		double c = sqrt(abs(r1*r1 - a*a));
		Vector V = (O2-O1).unit();
		Point H = O1 + V * a;
		X.push_back(H + V.ort() * c);
		if(c > EPS) X.push_back(H - V.ort() * c);
	}
	return X;
}

// LINEA AB vs CIRCULO (O, r)
// 1. Mucha perdida de precision, reemplazar por resultados de formula.
// 2. Considerar line o segment

vector <Point> lineCircleIntersection(Point A, Point B, Point O, long double r)
{
	vector <Point> X;
	Point H1 = O + (B - A).ort() * cross(O - A, B - A) / (B - A).mod2();
	long double d2 = cross(O - A, B - A) * cross(O - A, B - A) / (B - A).mod2();
	if(d2 <= r*r + EPS)
	{
		long double k = sqrt(abs(r * r - d2));
		Point P1 = H1 + (B - A) * k / (B - A).mod();
		Point P2 = H1 - (B - A) * k / (B - A).mod();
		if(between(A, B, P1)) X.push_back(P1);
		if(k > EPS && between(A, B, P2)) X.push_back(P2);
	}
	return X;
}
/*
// My own version
vector< Point > lineCircleIntersection( Point A , Point B , Point O , ld R ){
	Vector V = ( B - A ).ort();
	Point H = lineIntersection( A , B , O , O + V );
	if( R < ( O - H ).dist() - EPS ) return vector< Point >();
	if( abs( R - ( O - H ).dist() ) < EPS ) return vector< Point >( 1 , H );
	vector< Point > ans;
	Point P , Q;
	Vector W = ( B - A ).unit();
	ld d = sqrt( R*R - ( O - H ).dist2() );
	P = H + W*d , Q = H - W*d;
	ans.pb( P ) , ans.pb( Q );
	return ans;
}
*/
//### PROBLEMAS BASICOS ###############################################################
void CircumscribedCircle()
{
	int x1, y1, x2, y2, x3, y3;
	scanf("%d %d %d %d %d %d", &x1, &y1, &x2, &y2, &x3, &y3);
	Point A(x1, y1), B(x2, y2), C(x3, y3);
	Point P1 = (A + B) / 2.0;
	Point P2 = P1 + (B-A).ort();
	Point P3 = (A + C) / 2.0;
	Point P4 = P3 + (C-A).ort();
	Point CC = lineIntersection(P1, P2, P3, P4);
	double r = dist(A, CC);
	printf("(%.6lf,%.6lf,%.6lf)\n", CC.x, CC.y, r);
}
void InscribedCircle()
{
	int x1, y1, x2, y2, x3, y3;
	scanf("%d %d %d %d %d %d", &x1, &y1, &x2, &y2, &x3, &y3);
	Point A(x1, y1), B(x2, y2), C(x3, y3);
	Point AX = A + (B-A).unit() + (C-A).unit();
	Point BX = B + (A-B).unit() + (C-B).unit();
	Point CC = lineIntersection(A, AX, B, BX);
	double r = abs(area(A, B, CC) / dist(A, B));
	printf("(%.6lf,%.6lf,%.6lf)\n", CC.x, CC.y, r);
}
vector <Point>  TangentLineThroughPoint(Point P, Point C, long double r)
{
	vector <Point> X;
	long double h2 = (C - P).mod2();
	if(h2 < r*r) return X;
	else
	{
		long double d = sqrt(h2 - r*r);
		long double m1 = (r*(P.x - C.x) + d*(P.y - C.y)) / h2;
		long double n1 = (P.y - C.y - d*m1) / r;
		long double n2 = (d*(P.x - C.x) + r*(P.y - C.y)) / h2;
		long double m2 = (P.x - C.x - d*n2) / r;
		X.push_back(C + Point(m1, n1)*r);
		if(d != 0) X.push_back(C + Point(m2, n2)*r);
		return X;
	}
}
void TangentLineThroughPoint()
{
	int xc, yc, r, xp, yp;
	scanf("%d %d %d %d %d", &xc, &yc, &r, &xp, &yp);
	Point C(xc, yc), P(xp, yp);
	double hyp = dist(C, P);
	if(hyp < r) printf("[]\n");
	else
	{
		double d = sqrt(hyp * hyp - r*r);
		double m1 = (r*(P.x - C.x) + d*(P.y - C.y)) / (r*r + d*d);
		double n1 = (P.y - C.y - d*m1) / r;
		double ang1 = 180 * atan(-m1/n1) / PI + EPS;
		if(ang1 < 0) ang1 += 180.0;
		double n2 = (d*(P.x - C.x) + r*(P.y - C.y)) / (r*r + d*d);
		double m2 = (P.x - C.x - d*n2) / r;
		double ang2 = 180 * atan(-m2/n2) / PI + EPS;
		if(ang2 < 0) ang2 += 180.0;
		if(ang1 > ang2) swap(ang1, ang2);		
		if(d == 0) printf("[%.6lf]\n", ang1);
		else printf("[%.6lf,%.6lf]\n", ang1, ang2);
	}
}
void CircleThroughAPointAndTangentToALineWithRadius()
{
	int xp, yp, x1, y1, x2, y2, r;
	scanf("%d %d %d %d %d %d %d", &xp, &yp, &x1, &y1, &x2, &y2, &r);
	Point P(xp, yp), A(x1, y1), B(x2, y2);
	Vector V = (B - A).ort() * r / (B - A).mod();
	Point X[2];
	int cnt = 0;
	Point H1 = P + (B - A).ort() * cross(P - A, B - A) / (B - A).mod2() + V;
	double d1 = abs(r + cross(P - A, B - A) / (B - A).mod());
	if(d1 - EPS <= r)
	{
		double k = sqrt(abs(r * r - d1 * d1));
		X[cnt++] = Point(H1 + (B - A).unit() * k);	
		if(k > EPS) X[cnt++] = Point(H1 - (B - A).unit() * k);
	}
	Point H2 = P + (B - A).ort() * cross(P - A, B - A) / (B - A).mod2() - V;
	double d2 = abs(r - cross(P - A, B - A) / (B - A).mod());
	if(d2 - EPS <= r)
	{
		double k = sqrt(abs(r * r - d2 * d2));
		X[cnt++] = Point(H2 + (B - A).unit() * k);
		if(k > EPS) X[cnt++] = Point(H2 - (B - A).unit() * k);		
	}
	sort(X, X + cnt);
	if(cnt == 0) printf("[]\n");
	else if(cnt == 1) printf("[(%.6lf,%.6lf)]\n", X[0].x, X[0].y);
	else if(cnt == 2) printf("[(%.6lf,%.6lf),(%.6lf,%.6lf)]\n", X[0].x, X[0].y, X[1].x, X[1].y);
}
void CircleTangentToTwoLinesWithRadius()
{
	int x1, y1, x2, y2, x3, y3, x4, y4, r;
	scanf("%d %d %d %d %d %d %d %d %d", &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &r);
	Point A1(x1, y1), B1(x2, y2), A2(x3, y3), B2(x4, y4);
	Vector V1 = (B1 - A1).ort() * r / (B1 - A1).mod();
	Vector V2 = (B2 - A2).ort() * r / (B2 - A2).mod();
	Point X[4];
	X[0] = lineIntersection(A1 + V1, B1 + V1, A2 + V2, B2 + V2);
	X[1] = lineIntersection(A1 + V1, B1 + V1, A2 - V2, B2 - V2);
	X[2] = lineIntersection(A1 - V1, B1 - V1, A2 + V2, B2 + V2);
	X[3] = lineIntersection(A1 - V1, B1 - V1, A2 - V2, B2 - V2);
	sort(X, X + 4);
	printf("[(%.6lf,%.6lf),(%.6lf,%.6lf),(%.6lf,%.6lf),(%.6lf,%.6lf)]\n", X[0].x, X[0].y, X[1].x, X[1].y, X[2].x, X[2].y, X[3].x, X[3].y);
}

void CircleTangentToTwoDisjointCirclesWithRadius()
{
	int x1, y1, r1, x2, y2, r2, r;
	scanf("%d %d %d %d %d %d %d", &x1, &y1, &r1, &x2, &y2, &r2, &r);
	Point A(x1, y1), B(x2, y2);
	r1 += r;
	r2 += r;
	double d = dist(A, B);
	if(d > r1 + r2 || d < max(r1, r2) - min(r1, r2)) printf("[]\n");
	else
	{
		double a = (r1*r1 - r2*r2 + d*d) / (2.0*d);
		double b = d - a;
		double c = sqrt(abs(r1*r1 - a*a));
		Vector V = (B-A).unit();
		Point H = A + V * a;
		Point P1 = H + V.ort() * c;
		Point P2 = H - V.ort() * c;
		if(P2 < P1) swap(P1, P2);
		if(P1 == P2) printf("[(%.6lf,%.6lf)]\n", P1.x, P1.y);
		else printf("[(%.6lf,%.6lf),(%.6lf,%.6lf)]\n", P1.x, P1.y, P2.x, P2.y);	
	}
}
############################# Rotating Callipers ###############################################
vector< pair< pair< Point , Point > , Vector > > AntipodalPairs( vector< Point > P )
{
	int nP = P.size();
	vector< pair< pair< Point , Point > , Vector > > V;
	for( int i = 0 , j = 2 ; i < nP ; ++i ){
		// P[j] debe ser el punto mas lejano a la linea P[i], P[(i+1)%n]:		
		while(area(P[i], P[(i+1)%nP], P[(j+1)%nP]) > area(P[i], P[(i+1)%nP], P[j])) j = (j+1)%nP;
		// Par antipodal: i, j
		// Par antipodal: (i+1)%n, j		
		V.push_back( make_pair( make_pair( P[i] , P[j] ) , P[(i+1)%nP] - P[i] ) );
	}       
	return V;
}
///UVA 12307 - Smallest Enclosing Rectangle
// O(n)
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)v.size())

#define EPS (1e-8)
typedef long double ld;
bool equals( ld x , ld y ){ return abs( x - y ) < EPS; }
bool Less( ld x , ld y ){ return x < y - EPS; }
bool LessEqual( ld x , ld y ){ return x < y + EPS; }
struct Point{
	ld x , y;
	Point(){}
	Point( ld x , ld y ) : x( x ) , y( y ) {}
	Point ort(){ return Point( -y , x ); }
	ld dist(){ return sqrt( abs( x * x + y * y ) ); }
};
typedef Point Vector;
Point operator -( const Point &A , const Point &B ){ return Point( A.x - B.x , A.y - B.y );}
Point operator +( const Point &A , const Point &B ){ return Point( A.x + B.x , A.y + B.y );}
Point operator *( const Point &A , const ld &k ){ return Point( A.x * k , A.y * k );}
bool operator < ( const Point &A , const Point &B ){
	if( !equals( A.x , B.x ) ) return Less( A.x , B.x );
	return Less( A.y , B.y );
}
ld dist( const Point &A , const Point &B ){ return ( B - A ).dist();}
ld cross( Point A , Point B ){ return A.x * B.y - A.y * B.x;}
ld area( Point A , Point B , Point C ){ return cross( B - A , C - A ); }

Point lineIntersection( const Point &A , const Point &B , const Point &C , const Point &D ){ 
	return A + ( B - A ) * ( cross( C - A , D - C ) / cross( B - A , D - C ) );
}
bool parallel( Vector A , Vector B ){
	return equals( cross( A , B )  , 0.0 );
}
ld proyection( Point A , Point B , Point C ){
	Point I = lineIntersection( A , B , C , C + (B - A).ort() );
	bool op1 = area( A , A + (B - A).ort() , B ) > 0;
	bool op2 = area( A , A + (B - A).ort() , I ) > 0;
	if( op1 ^ op2 ) return -dist( I , A );
	return dist( I , A );
}
ld high( Point A , Point B , Point C ){
	Point I = lineIntersection( A , B , C , C + (B - A).ort() );
	return dist( I , C );
}
vector< Point > convexHull( vector< Point > P ){
	Point H[ SZ(P) << 1 ];
	sort( all( P ) );
	int sz = 0;
	REP( i , SZ( P ) ){
		while( sz >= 2 && area( H[ sz - 2 ] , H[ sz - 1 ] , P[ i ] ) < EPS ) sz --;
		H[ sz ++ ] = P[ i ];
	}
	for( int i = SZ( P ) - 2 , m = sz ; i >= 0 ; --i ){
		while( sz > m && area( H[ sz - 2 ] , H[ sz - 1 ] , P[ i ] ) < EPS ) sz --;
		H[ sz ++ ] = P[ i ];
	}
	return vector< Point > ( H , H + sz - 1 );
}
int n;
int next( int p ){ return (p == n - 1) ? 0 : (p + 1);}
int prev( int p ){ return p == 0 ? (n - 1) : (p - 1);}
int main(){
	while( sc( n ) == 1 ){
		if( !n ) break;
		vector< Point > P;
		REP( i , n ){
			double x , y;
			scanf( "%lf %lf" , &x , &y );
			P.pb( Point( x , y ) );
		}
		P = convexHull( P );
		n = SZ( P );

		ld ans1 = 1e100 , ans2 = 1e100;

		for( int p = 0 , q = 1 , r = 1 , t = -1 ; p < n ; ++p ){
			while( LessEqual( high( P[ p ] , P[ next( p ) ] , P[ q ] ) , high( P[ p ] , P[ next( p ) ] ,  P[ next( q ) ] ) ) ) q = next( q );
			if( t == -1 ) t = q;
			while( LessEqual( proyection( P[ p ] , P[ next( p ) ] , P[ r ] ) , proyection( P[ p ] , P[ next( p ) ] ,  P[ next( r ) ] ) ) )
				r = next( r );
			while( LessEqual( proyection( P[ p ] , P[ next( p ) ] , P[ next( t ) ] ) , proyection( P[ p ] , P[ next( p ) ] ,  P[ t ] ) ) )
				t = next( t );

			Point A , B , C;
			Vector V = P[ next( p ) ] - P[ p ];
			Vector W = V.ort();

			A = lineIntersection( P[ p ] , P[ p ] + V , P[ t ] , P[ t ] + W );
			B = lineIntersection( P[ p ] , P[ p ] + V , P[ r ] , P[ r ] + W );
			C = lineIntersection( P[ q ] , P[ q ] + V , P[ r ] , P[ r ] + W );
			ans1 = min( ans1 , abs( area( A , B , C ) ) );
			ans2 = min( ans2 , abs( dist( A , B ) + dist( B , C ) ) );

		}
		printf( "%.2f %.2f\n" , (double)ans1 , (double)(2.0*ans2) );
	}
}
////////////////////////////////////////////
//6402_ACM
// circle polygon intersection
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define N 100
#define INF (1e6)
#define EPS (1e-8)

typedef long double ld;
typedef long long ll;
typedef unsigned long long ull;

const ld PI = acos( -1.0 );
bool equals( const ld &x , const ld &y ){ return abs( x - y ) < EPS;}
inline bool Less( const ld &x , const ld &y ){ return x < y - EPS;}
struct Point{
	ld x , y;
	Point(){ x = y = 0;}
	Point( ld x , ld y ) : x( x ) , y( y ){}
	Point ort(){ return Point( -y , x );}
	ld dist(){ return hypot( x , y );}
	ld dist2(){ return x * x + y * y ;}
	ld arg(){ ld t = atan2( y , x ); return equals( t , PI )?(-PI):t;}
	Point unit(){ ld r = dist(); return equals( r , 0 ) ? Point() : Point( x / r , y / r );}
};
typedef Point Vector;
Point operator - ( const Point &A , const Point &B ){ return Point( A.x - B.x , A.y - B.y );}
Point operator + ( const Point &A , const Point &B ){ return Point( A.x + B.x , A.y + B.y );}
Point operator * ( const Point &A , const ld &K ){ return Point( A.x * K , A.y * K );}
ld cross( const Point &A , const Point &B ){ return A.x * B.y - A.y * B.x;}
ld area( const Point &A , const Point &B , const Point &C ){ return cross( B - A , C - A );}
Point lineIntersection( const Point &A , const Point &B , const Point &C , const Point &D ){
	return A + ( (B - A)* ( cross( C - A , D - C ) / cross( B - A , D - C ) ));
}
bool between( const Point &A , const Point &B , const Point &P ){
	return 	min( A.x , B.x ) < P.x + EPS && P.x < max( A.x , B.x ) + EPS &&
			min( A.y , B.y ) < P.y + EPS && P.y < max( A.y , B.y ) + EPS;
}
bool onSegment( const Point &A , const Point &B , const Point &P ){
	return abs( area( A , B , P ) ) < EPS && between( A , B , P );
}

vector< Point > lineCircleIntersection( Point A , Point B , Point OO , ld RR ){
	Vector V = ( B - A ).ort();
	Point H = lineIntersection( A , B , OO , OO + V );
	if( RR < ( OO - H ).dist() - EPS ) return vector< Point >();
	if( abs( RR - ( OO - H ).dist() ) < EPS ) return vector< Point >( 1 , H );
	vector< Point > ans;
	Point P , Q;
	Vector W = ( B - A ).unit();
	ld d = sqrt( RR*RR - ( OO - H ).dist2() );
	P = H + W*d , Q = H - W*d;
	ans.pb( P ) , ans.pb( Q );
	return ans;
}
int n , rad;
Point P[ N + 5 ], O;
ld R , S;
ld solveParticular( Point A , Point B ){
	ld tA = A.arg() , tB = B.arg();
	if( equals( tA , tB ) ) return 0;
	ld t;
	if( tA < tB ) t = tB - tA;
	else t = PI - tA + (tB - (-PI) );
	ld op1 = area( O , A , B );
	ld op2 = (t * R * R);
	ld ans = min( op1 , op2 );
	return ans;
}
ld solveGeneral( Point A , Point B ){
	ld tA = A.arg() , tB = B.arg();
	if( equals( tA , tB ) ) return 0;
	ld sign = 1;
	if( Less( area( O , A , B ) , 0 ) ) {
		sign = -1;
		swap( A , B );
	}
	vector< Point > vec = lineCircleIntersection( A , B , O , R ) , vec2;
	REP( i , SZ( vec ) )
		if( onSegment( A , B , vec[ i ] ) ) vec2.pb( vec[ i ] );
	if( SZ( vec2 ) == 2 ){
		if( !Less( area( O , A , vec2[ 0 ] ) , area( O , A , vec2[ 1 ] ) ) ) swap( vec2[ 0 ] , vec2[ 1 ] );
	}
	vec2.insert( vec2.begin() , A );
	vec2.pb( B );
	ld ans = 0;
	REP( i , SZ( vec2 ) - 1 )
		ans += solveParticular( vec2[ i ] , vec2[ i + 1 ] );
	return sign * ans;
}
inline int next( int p ){ return p == n - 1 ? 0 : (p + 1);}
ld f(){
	ld ans = 0;
	REP( i , n ) ans += solveGeneral( P[ i ] , P[ next( i ) ] );
	ans = abs( ans );
	return ans/2.0;
}
int main(){
	int x , y;
	while( sc( n ) == 1 ){
		sc( rad );
		REP( i , n ){
			sc( x ) , sc( y );
			P[ i ] = Point( x , y );
		}
		R = rad;
		printf( "%.10f\n" , (double) f() );
	}
}


// smallest enclosing parallelogram O(n^2)
#-----------------------------------------------------------------#
############################# Spherical Geometry ###############################################
//http://en.wikipedia.org/wiki/Geographic_coordinate_system
// Great-circle_distance
//http://en.wikipedia.org/wiki/Great-circle_distance
//535_UVA
//a = latitude , b = longitude 
ld cost = sin( a1 )*sin( a2 ) + cos( a1 )*cos( a2 )*cos( (b1 - b2) );
############################# Rotation Matrix ###############################################
/// 11507_UVA 3D
Vector rot( Vector V , ld senA , ld cosA )
{
	return Point( V.x*cosA - V.y*senA , V.x*senA + V.y*cosA );
}
http://en.wikipedia.org/wiki/Rotation_matrix

#----------------------------- SWEEP LINE ------------------------------------#
#------------------------------UNION DE INTERVALOS-----------------------------------#
//Union de intervalos (cuadriculas) (nlogn)
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ(v) ((int)v.size())

#define mp make_pair
#define fi first
#define se second

#define M 1000000

#define test puts( "************test**************" );

typedef long long ll;
typedef vector< int > vi;
typedef pair< int , int > pii;
typedef vector< pii > vpii;

#define v1 ((node<<1) + 1)
#define v2 (v1+1)
#define med ((a+b)>>1)
#define LEFT v1 , a , med
#define RIGHT v2 , med + 1 , b
struct Node{
   	int mini, freq , flag;
   	Node(){ flag = 0 , mini = 1e9 , freq = 0; }
	Node( int mini , int freq , int flag ) : mini( mini ) , freq( freq ) , flag( flag ) {}
};
Node operator + ( const Node &a , const Node &b ){
	if( a.mini != b.mini ){
		if( a.mini < b.mini ) return a;
		return b;
	}
	return Node( a.mini , a.freq + b.freq , 0 );
}
struct ST{
	vector< Node >T;
	int n , total;
	ST(){ T.resize( 500 ); }
	/*
	ST( vi &v ){
		n = SZ( v );
		total = 0;
		REP( i , n ) total += v[ i ];
		T.resize( 4*n );
		build_tree( v , 0 , 0 , n - 1 );
	}
	*/
	void init( vi &v ){
		n = SZ( v );
		total = 0;
		REP( i , n ) total += v[ i ];
		build_tree( v , 0 , 0 , n - 1 );
	}
	
	void build_tree( vi &v , int node , int a ,  int b ){
		if( a == b ){
			T[ node ] = Node( 0 , v[ a ] , 0 );
			return;
		}
		build_tree( v , LEFT ) , build_tree( v , RIGHT );
		T[ node ] = T[ v1 ] + T[ v2 ];
	}
	void push( int node , int a , int b ){
		if( T[ node ].flag == 0 ) return ;
		T[ node ].mini += T[ node ].flag ;
		if( a != b ){
			T[ v1 ].flag += T[ node ].flag;
			T[ v2 ].flag += T[ node ].flag;
		}
		T[ node ].flag = 0;
	}
	int area(){ 
		push( 0 , 0 , n - 1 );
		return T[ 0 ].mini == 0 ? ( total - T[ 0 ].freq ) : total ;
	}
	void update ( int node , int a , int b , int lo , int hi , int val ){
		push( node , a , b );
		if( lo > b || a > hi ) return;
		if( a >= lo && hi >= b ) {
			T[ node ].flag = val;
			push( node , a , b );
			return;	
		}
		update( LEFT , lo , hi , val );
		update( RIGHT , lo , hi , val );
		T[ node ] = T[ v1 ] + T[ v2 ];
	}
	void update( int lo , int hi , int val ){
		update( 0 , 0 , n - 1 , lo , hi , val );
	}
}st;

struct event{
	int x , t , lo , hi;
	event(){}
	event( int x , int t , int lo , int hi ) :x( x ) , t( t ) , lo( lo ) , hi( hi ) {}
};
bool operator < ( const event &a , const event &b ){
	if( a.x != b.x ) return a.x < b.x;
	return a.t < b.t;
}
struct rect{
	ll x1 , x2 , y1 , y2;
	rect(){}
	rect( ll x1 , ll x2 , ll y1 , ll y2 ) : x1( x1 ) , x2( x2 ) , y1( y1 ) , y2( y2 ) {}
	void impr(){ cout << "(" << x1 << "," << x2 << ")  (" << y1 << "," << y2 << ")" << endl;}
};

bool intersection( rect A , rect B , rect &C ){
	if( A.x1 > B.x1 ) swap( A , B );
	if( B.x1 > A.x2 ) return 0;
	C.x1 = B.x1 , C.x2 = min( A.x2 , B.x2 );
	if( A.y1 > B.y1 ) swap( A , B );
	if( B.y1 > A.y2 ) return 0;
	C.y1 = B.y1 , C.y2 = min( A.y2 , B.y2 );
	return 1;
}

int getInd( vi &v , int x ){
	return lower_bound( all( v ) , x ) - v.begin();
}
void impr( vi &v ){
	REP( i , SZ( v ) ) printf( "%d%c" , v[ i ] , i + 1 == SZ( v ) ? 10 : 32 );
}
ll unionRect( vector< rect > rectangles ){
	if( SZ( rectangles ) == 0 ) return 0;
	vector< event > E;
	vi L;
	REP( i , SZ( rectangles ) ){
		rect r = rectangles[ i ];
		int x1 = r.x1 , y1 = r.y1 , x2 = r.x2 , y2 = r.y2;
		E.pb( event( x1 , 0 , y1 , y2 ) );
		E.pb( event( x2 + 1 , 1 , y1 , y2 ) );
		L.pb( y1 ) , L.pb( y2 );
	}
	sort( all( L ) );
	L.resize( unique( all( L ) ) - L.begin() );
	vi v;
	v.pb( 1 );
	for( int i = 1 ; i < SZ( L ) ; ++i ){
		v.pb( L[ i ] - L[ i - 1 ] - 1 );
		v.pb( 1 );
	}
	st.init( v );
    ll ans = 0;
    sort( all( E ) );
    REP( i , SZ( E ) ){
		int ind;
		for( int j = i ; j < SZ( E ) ; ++j )
			if( E[ j ].x == E[ i ].x ) ind = j;
			else break;
		int delta = 0;
		if( ind + 1 < SZ( E ) ) delta = E[ ind + 1 ].x - E[ ind ].x;
		
		for( int j = i ; j <= ind ; ++j ){
			int lo = getInd( L , E[ j ].lo ) << 1;
			int hi = getInd( L , E[ j ].hi ) << 1;
			st.update( lo , hi , E[ j ].t == 0 ? +1 : -1 );
		}
		ans += (ll)st.area() * (ll)delta;
		i = ind;
	}
    return ans;
};

int n;
int X[ M + 5 ] , Y[ M + 5 ];
ll get( int timer  , ll t , ll R , ll C ){
	vector< rect > rectangles;
	ll x = t/C , y = t%C;
	rect a( -1 , x - 1 , -1 , C - 1 );
	rect b( x , x , -1 , y );
	rect c( 0 , R - 1 , 0 , C - 1 );
	rect s;
	rect r;

	REP( i , n ){
		int x = X[ i ] , y = Y[ i ];
		rect A( x - timer , x + timer , y - timer , y + timer );
		if( intersection( A , a , r ) ){
			if( intersection( r , c , r ) ){
				rectangles.pb( r );
			}
		}
		if( intersection( A , b , r ) ){
			if( intersection( r , c , r ) ){
				rectangles.pb( r );
			}
		}
	}
	return unionRect( rectangles );
}
//Union de intervalos (cuadriculas) (n^2)
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ(v) ((int)v.size())

#define mp make_pair
#define fi first
#define se second

#define M 1000000

#define test puts( "************test**************" );

typedef long long ll;
typedef vector< int > vi;
typedef pair< int , int > pii;
typedef vector< pii > vpii;


struct rect{
	ll x1 , x2 , y1 , y2;
	rect(){}
	rect( ll x1 , ll x2 , ll y1 , ll y2 ) : x1( x1 ) , x2( x2 ) , y1( y1 ) , y2( y2 ) {}
	void impr(){ cout << "(" << x1 << "," << x2 << ")  (" << y1 << "," << y2 << ")" << endl;}
};

bool intersection( rect A , rect B , rect &C ){
	if( A.x1 > B.x1 ) swap( A , B );
	if( B.x1 > A.x2 ) return 0;
	C.x1 = B.x1 , C.x2 = min( A.x2 , B.x2 );
	if( A.y1 > B.y1 ) swap( A , B );
	if( B.y1 > A.y2 ) return 0;
	C.y1 = B.y1 , C.y2 = min( A.y2 , B.y2 );
	return 1;
}

void impr( vi &v ){
	REP( i , SZ( v ) ) printf( "%d%c" , v[ i ] , i + 1 == SZ( v ) ? 10 : 32 );
}
int getInd( vi &v , int x ){
	return lower_bound( all( v ) , x ) - v.begin();
}
ll unionRect( vector< rect > rectangles ){
	if( SZ( rectangles ) == 0 ) return 0;
    ll ans = 0;
    vi x , y;
    REP( i , SZ( rectangles ) ) {
		x.pb( rectangles[ i ].x1 );
		x.pb( rectangles[ i ].x2 + 1 );
		y.pb( rectangles[ i ].y1 );
		y.pb( rectangles[ i ].y2 + 1 );
	}
	sort( all( x ) );
	sort( all( y ) );
	x.resize( unique( all( x ) ) - x.begin() );
	y.resize( unique( all( y ) ) - y.begin() );
	
	REP( i , SZ( x ) - 1 ){
		vi T( SZ( y ) + 1 );
		REP( k , SZ( rectangles ) ){
			if( x[ i ] >= rectangles[ k ].x1 && x[ i + 1 ] - 1 <= rectangles[ k ].x2 ){
				T[ getInd( y , rectangles[ k ].y1 ) ] ++;
				T[ getInd( y , rectangles[ k ].y2 + 1 ) ] --;
			}
		}
		int sum = 0;
		REP( j , SZ( y ) ){
			sum += T[ j ];
			if( sum > 0 ) ans += (ll)( x[ i + 1 ] - x[ i ] ) * (ll)( y[ j + 1 ] - y[ j ] );
		}
	}
    return ans;
};

int n;
int X[ M + 5 ] , Y[ M + 5 ];
ll get2( int timer , ll R , ll C ){
	vector< rect > rectangles;
	rect c( 0 , R - 1 , 0 , C - 1 );
	rect s;
	rect r;
	REP( i , n ){
		int x = X[ i ] , y = Y[ i ];
		rect A( x - timer , x + timer , y - timer , y + timer );
		if( intersection( A , c , r ) ){
			rectangles.pb( r );
		}
	}
	return unionRect( rectangles );
}
////////////////////////////// Largest empty Rectangle  //////////////////////////////
//UVA 12830
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

typedef long long ll;
typedef pair< int , int > pii;
typedef vector< int > vi;
typedef vector< vi > vvi;

const int maxn = 10010;  
  
int h[maxn], l[maxn], r[maxn];  
int n, m, ans;  
map<int, vector<int> > tree;  
  
void check() {  
    for (int i = 0, j = n; i <= n; i++, j--) {  
        for(l[i] = i; l[i] > 0 && h[l[i]-1] >= h[i]; )  
            l[i] = l[l[i]-1];  
        for(r[j] = j; r[j] < n && h[r[j]+1] >= h[j]; )  
            r[j] = r[r[j]+1];  
    }  
}  
  
void cal() {  
    for (int i = 0; i <= n; i++) {  
        int tmp = h[i] * (r[i] - l[i] + 2);  
        ans = max(ans, tmp);  
    }  
}  
  
int main() {  
    int cases;  
	sc( cases );
	REP( tc , cases ){  
        tree.clear();  
		sc( n ) , sc( m );
        int op, x, y, dx, dy; 
        int cnt ;
        sc( cnt );
        REP( i , cnt ){  
			sc( x ) , sc( y );
            tree[ y ].push_back( x );  
        }  
        tree[ m ];
        ans = max(n, m);  
        int last = 0;  
		clr( h , 0 );
        map< int , vi >::iterator it;  
        for( it = tree.begin(); it != tree.end(); it++ ){  
            int d = it->first - last;
            last += d;  
            for (int i = 1; i < n; i++)  
                h[ i ] += d;  
            check();
            cal();  
            vi tmp = it->second;  
            REP( i , SZ( tmp ) ) h[ tmp[ i ] ] = 0;  
        }  
        printf("Case %d: %d\n", tc + 1 , ans );  
    }  
} 

##########################  JAVA - BIGINTEGER #################################
////En implementacion
Variables globales en java
{
	Examples:
	static int phi[] = new int [MAXN + 1];
    static BigInteger K[] = new BigInteger[ 605 ];
    Obs 
    - no olvidar el static ya que esta sentencia permite el poder usar la variable desde otro lado del code
	-para el primero como int es un dato primitivo no es necesario crear referencias por cada un de los int creados 
    pero para el segundo si es necesario para evitar excepciones
    for( int i = 1 ; i < 604 ; ++i ){
            K[ i ] = BigInteger.ONE;
            K[ i ] = K[ i - 1 ].multiply( kk );
    }
}
Constructores
{
	String s;
	BigInteger num = BigInteger(s);
	int k;
	BigInteger num = BigInteger.valeOf(k);
}

Operaciones aritmeticas
{
	BigInteger ans = BigInteger;
	ans = num.subtract(den);
	ans = num.multiply(den));
	ans = num.add(den);
	ans = num.divide(den);	
	REVERSE_MULTIPLIER = BigInteger.valueOf(MULTIPLIER).modInverse(BigInteger.valueOf( mod )).longValue();// modInverse , longValue()
}
Operaciones con Bigdecimal
{
	System.out.printf(Locale.US,"%.10f\n", s);//Para que no se contraiga con la notacion cientifica ejm : 1.8E-10
	.toPlainString();	//Returns a string representation of this BigDecimal without an exponent field.
	stripTrailingZeros()	// Returns a BigDecimal which is numerically equal to this one but with any trailing zeros removed from the representation. ejm 5^2 == 25.00 > 25
}
String en Java
	Acceso a elementos s.charAt(i);
	Tama\F1o s.length();
	Substring en el intervalo [a-b> : s.substring(a, b);
	Substring en el intervalo [a-oo> : s.substring(a);
	//INVESTIGAR s.endsWith("ize") s.startsWith("post")
	///OJO INVESTIGAR (PARECE s.find )cad.indexOf('+')) != -1
Scanner I/O en JAVA
	Declaracion 	Scanner cin = new Scanner(System.in);
	String s = cin.nextLine();// next();
Parsing
	String s;
	int t = Integer.parseInt( s );	
Map en JAVA
	Map< String , Integer >M = new HashMap<String, Integer>();
      
	if( !M.containsKey(temp) )
	   M.put(temp, 1 );
	else M.put(temp, M.get(temp) + 1 );
    for (Map.Entry<String, Integer> entry : M.entrySet())
    {
        max = entry.getValue();
        ans = entry.getKey();
    }
Set en java // Array de Set
	TreeSet< obj >[] set = new TreeSet [ ns + 1 ];
    for( int i = 0 ; i <= ns ; ++i ) set[ i ] = new TreeSet< obj >();
Objects en java
	new // siempre usarlo al crear un nuevo objeto ya que a modo practico esta sentencia como que creara espacio fisico en la variable
	Example : set[ 2*k ].add( new obj( H1.hash( i , i + k + 1 ) , i ) );
	Objeto comparador 
	public static class obj implements Comparable< obj > {
        long a ;
        int b;

        obj(long a, int b) {
            this.a = a;
            this.b = b;
        }
        public int compareTo( obj o )
        {
            if( o.a != a )return (int)(a - o.a);
            return (int)( cc.charAt( b ) - cc.charAt( o.b ) );
        }
    }
	Tambien se puede hacer este truco //http://codeforces.com/contest/19/submission/86646
	Integer o[] = new Integer[ sz ];
    for( int i = 0 ; i < sz ; ++i ) o[ i ] = i;
    Arrays.sort( o , new Comparator< Integer >() {
        public int compare( Integer o1 , Integer o2 )
        {
            if( H1[ o1 ] != H1[ o2 ] ) return (int)( H1[ o2 ] - H1[ o1 ] );
            return (int)( H2[ o2 ] - H2[ o1 ] );
        }    
    });
    /*http://codeforces.com/contest/19/submission/86646
Integer o[] = new Integer[ sz ];
for( int i = 0 ; i < sz ; ++i ) o[ i ] = i;
Arrays.sort( o , new Comparator< Integer >() {
    public int compare( Integer o1 , Integer o2 )
    {
        if( H1[ o1 ] != H1[ o2 ] ) return (int)( H1[ o2 ] - H1[ o1 ] );
        return (int)( H2[ o2 ] - H2[ o1 ] );
    }    
});
*/
// StringHash taked of EgorK , http://codeforces.com/contest/113/submission/676303
// Pendiente arraylist y otros coleccions
// Sorting array  Arrays.sort( A , 0 , n - 1 ); creo xd
// Freopen in java

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.Scanner;

public class Main {
    public static String input = "toral.in";
    public static String output = "toral.out";
    
    public static void main(String[] args) throws FileNotFoundException {
       Scanner cin = new Scanner( new File(input) );
       BigInteger a = cin.nextBigInteger();
       BigInteger b = cin.nextBigInteger();
       cin.close();
       PrintWriter s = new PrintWriter(output);
       s.println( a.gcd( b ) );
       s.close();
    }
}

//
public int compareTo(BigInteger val)
Parameters
val - BigInteger to which this BigInteger is to be compared

Return Value
This method returns -1, 0 or 1 as this BigInteger is numerically less than, equal to, or greater than val.

//
Nextpermutation in JAVA	
int[] nextperm(int[] vals) {
   int i =  vals.length-1;
   while (true) {
      int ii =  i;
      i--;
      if (vals[i] < vals[ii]) {
         int j = vals.length;
         while (vals[i] >= vals[--j]);
            int temp = vals[i];  //Swap
            vals[i] = vals[j]; 
            vals[j] = temp;
         int begin = ii, end = vals.length-1;
         while (end>begin) {
               int stemp = vals[end];   //Swap
               vals[end] = vals[begin]; 
               vals[begin] = stemp;
            end--; begin++;
         }   
         return vals;
      } else if (vals[i] == vals[0]) {
         int begin = 0, end = vals.length-1;
         while (end>begin) {
               int stemp = vals[end];   //Swap
               vals[end] = vals[begin]; 
               vals[begin] = stemp;
            end--; begin++;
         }   
         return vals;
      }
   }
}	
#-----------------------------------------------------------------#
########################## OTHERS #################################
########################## Perfect Hashing #################################
int f(string s){
	int id = 0;
	int n = s.size();
	for(int i = 0; i < n ; ++i){
		int val=0;
		for(int j = i + 1 ; j < n ; ++j)
			if(s[i]>s[j]) val++;
		id=id*(n-i)+val;					
	}
	return id;
}
#-----------------------------------------------------------------#
########################## Permutation #################################
string f(string s,long long N)
{
    if(N==1) return s;
    int n = s.size();
    for ( int i = 0 ; i < n ; ++i )
        if( (i+1)*fact[n-1]>=N ) return s[i]+f(s.substr(0,i)+s.substr(i+1),N-(fact[n-1])*(i));  
}
#-----------------------------------------------------------------#
###################################### BIG NUMS ###################################### 
/// ALERTA PROD puede ser cuadratico en la longitud
### SUMA DE CADENAS ################################################################## 
string suma(const string &a, const string &b)  
{  
    int LA = a.size(), LB = b.size(), L = max(LA, LB), carry = 0; 
    string x = string(L, '0');  
    while(L--) 
    { 
        LA--; LB--; 
        if(LA >= 0) carry += a[LA] - '0';  
        if(LB >= 0) carry += b[LB] - '0';      
        if(carry < 10) x[L] = '0' + carry, carry = 0; 
        else x[L] = '0' + carry - 10, carry = 1; 
    } 
    if(carry) x = '1' + x;  
    return x; 
}
//### PRODUCTO DE CADENAS ################################################################## 
string prod(string a, string b) 
{ 
    if(a=="0" || b=="0") return "0"; 
    else if(a.size()==1) 
    { 
        int m = a[0] - '0';
        string ans = string(b.size(), '0'); 
        int lleva = 0; 
        for(int i=b.size()-1; i>=0; i--) 
        { 
            int d = (b[i] - '0') * m + lleva; 
            lleva = d/10; 
            ans[i] += d%10; 
        } 
        if(lleva) ans = (char)(lleva + '0') + ans; 
        return ans; 
    } 
    else if(b.size()==1) return prod(b, a); 
    else
    { 
        string ans = "0"; 
        string ceros = ""; 
        for(int i=a.size()-1; i>=0; i--) 
        { 
            string s = prod(string(1, a[i]), b) + ceros; 
            ceros += "0"; 
            ans = suma(ans, s); 
        } 
        return ans; 
    } 
} 
bool cmp( string a , string b )
{
	if( a.size() == b.size() ) return a < b;
	return a.size() < b.size();
}
#-----------------------------------------------------------------#
########################## STABLE MARRIAGE #################################
// 243_SPOJ
//3837ACM
#define N 505
int n;
int pref_men[ N ][ N ] , pref_women[ N ][ N ] , inv[ N ][ N ] , cont[ N ] , husband[ N ] , wife[ N ];
void stable_marriage()
{
	REP( i , n )REP( j , n ) inv[ i ][ pref_women[ i ][ j ] ] = j;
	clr( cont , 0 );
	clr( husband , -1 );
	int m , w , dumped;
	REP( i , n )
	{
		m = i;
		while( m >= 0 ){
			while( 1 )
			{
				w = pref_men[ m ][ cont[ m ] ];
				++cont[ m ];
				if( husband[ w ] < 0 || inv[ w ][ m ] < inv[ w ][ husband[ w ] ] ) break;
			}
			dumped = husband[ w ];
			husband[ w ] = m;
			wife[ m ]=  w;
			m = dumped;
		}
	}
}
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
//Sudoku
#include<cstdio>
#include<cstring>
using namespace std;
 
bool resuelto;
 
int t[9][9];
 
bool Fil[9][10];
bool Col[9][10];
bool Cua[9][10];
 
void print(){
    for(int i=0;i<9;++i){
        for(int j=0;j<9;++j){
            printf("%d",t[i][j]);
        }
        puts("");
    }
}
 
void back(int fil,int col){
    if(resuelto==true){
        return;
    }
    
    if(fil==9){
        resuelto=true;
        print();
        return;
    }
    if(t[fil][col]>0){
        back(col<8?fil:fil+1,col<8?col+1:0);
        return;
    }
    int cua=(fil/3)*3+col/3;
    for(int i=1;i<=9;++i){
        if(Fil[fil][i]==true)continue;
        if(Col[col][i]==true)continue;
        if(Cua[cua][i]==true)continue;
        t[fil][col]=i;
        Fil[fil][i]=true;
        Col[col][i]=true;
        Cua[cua][i]=true;
        back(col<8?fil:fil+1,col<8?col+1:0);
        t[fil][col]=0;
        Fil[fil][i]=false;
        Col[col][i]=false;
        Cua[cua][i]=false;
    }
}
 
void caso(){
    resuelto=false;
    memset(Fil,false,sizeof(Fil));
    memset(Col,false,sizeof(Col));
    memset(Cua,false,sizeof(Cua));
    for(int i=0;i<9;++i){
        for(int j=0;j<9;++j){
            scanf("%1d",&t[i][j]);
            Fil[i][t[i][j]]=true;
            Col[j][t[i][j]]=true;
            Cua[(i/3)*3+j/3][t[i][j]]=true;
        }
    }
    //print();
    //puts("***************");
    back(0,0);
}
 
int main(){
    int n;
    scanf("%d",&n);
    for(int i=0;i<n;++i){
        caso();
    }
}
//Count Digits
//ACM 6527 - Counting ones
ll b = 2;
ll g(ll x ,ll d){
	ll r = 0;
    while( x > 0 ){
		if( x % b == d ) r++;
		x /= b;
    }
	return r;
}
ll f( ll N , ll D){
	if( N <= 0 ) return 0;
	ll U= N/b + (N%b >= D) - (D == 0);
	ll O = b * f( N/b - 1 ,D);
    for(ll i=(N/b)*b; i <= N ; i++ ) O += g( i/b ,D );
	return U+O;    
}
// Roman Numbers ( arabic to roman )
#define SIZE 13
string R[SIZE] = { "I" , "IV", "V" , "IX" , "X" , "XL" ,"L" , "XC","C" , "CD" , "D" , "CM" ,"M" };
int P[SIZE] = {     1,     4 ,   5 ,   9,    10 ,  40 ,  50 ,  90 , 100 , 400 , 500 , 900 , 1000 };
string g( int k )
{
	for( int i = 12 ; i >= 0 ; --i )
		if( P[i] == k ) return R[i];
		else if( P[i] < k )return R[i] + g( k - P[i] );
}
//( roman to arabic)
// 529_SRM_DIV1_250
int f( string s )
{
	int ns = s.size();
	int ans = 0;
	for( int i = 0 ; i < ns ; ++i )
	{
		if( i + 1  < ns )
		{
			for( int j = 12 ; j >= 0 ; --j )
			{
				string t = s.substr( i , R[j].size() );
				if( t == R[ j ] )
				{
					if( t.size() == 2 )i++;
					ans += P[ j ];
					break;
				}
			}
		}
		else
		{
			for( int j = 12 ; j >= 0 ; --j )
			{
				string t = s.substr( i , 1 );
				if( t == R[ j ] )
				{
					ans += P[ j ];
					break;
				}
			}
		}
	}
	return ans;
}
int f( string s )
{
	int ns = s.size();
	if( ns == 0 ) return 0;
	if( ns == 1 ){
		for( int i = 12 ; i  >= 0 ; --i )
			if( R[i].size() == 1 && R[i][0] == s[0] ) return P[i];
	}
	for( int i = 12 ; i  >= 0 ; --i )
		if( R[i].size() == 1 ){
			if( R[i][0] == s[0] ) return P[i] + f( s.substr( 1 ) );
		}
		else{
			if( R[i][0] == s[0] && R[i][1] == s[1] ) return P[i] + f( s.substr( 2 ) );
		}
}
//union de intervalos 
double f( vector< pair< double , double > > &A )
{
	int m = A.size();
	if( m == 0 )return 0;
	sort( all( A ) );
	double S = A[0].se - A[0].fi , end = A[0].se;
	for( int i = 1 ; i < m ; ++i )
	{
		if( A[i].fi > end ) S += A[i].se - A[i].fi;
		else if( A[i].se > end ) S += A[i].se - end;
		end = max( end , A[i].se );
	}
	return S/cte;
}
############################# TERNARY SEARCH ################################
// COMENTAR USO
// Ternary de punto minimo 
		double lo = -INF , hi = INF;
		int it = IT;
		while( it-- )
		{
			double med1 = (lo*2 + hi)/3 , med2 = (lo + hi*2)/3;
			if( f( med1 ) < f( med2 ) )
				hi = med2;
			else lo = med1;
		}
		printf( "%.3lf\n" , f(lo) );
#-----------------------------------------------------------------#
############################# CONVEXHULL TRICK ################################
// 6131_ACM
// 189 C DIV1 CDF
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define N 100005
#define EPS (1e-8)

typedef vector< int > vi;
typedef long long ll;
typedef long double ld;

struct Point{
    ll m , b;
    Point(){}
    Point( ll m , ll b ): m( m ) , b( b ) {}
    ld slope(){ return -1.0*b/m; }
};
Point operator - ( const Point &A , const Point &B ){ return Point( A.m - B.m , A.b - B.b ); }
int sz = 0;
Point H[ N + 5 ];
ll mm[ N + 5 ];
void add( ld m , ld b ){
	Point P( m , b );
    while( sz >= 2 && ( P - H[ sz - 1 ] ).slope() - ( H[ sz - 1 ] - H[ sz - 2 ] ).slope() < EPS ) sz--;
    H[ sz ] = P;
    if( sz ) mm[ sz - 1 ] = ( P - H[ sz - 1 ] ).slope();
    sz ++;
}
ll eval( ll x ){
	int ind = lower_bound( mm , mm + sz - 1 , (ld)x ) - mm;
	return x * H[ ind ].m + H[ ind ].b;
}
// when queries come in ascendent form
int pos;
ll eval( ll x ){
	while( pos < sz - 1 && mm[ pos ] < x ) pos ++;
	return x * H[ pos ].m + H[ pos ].b;
}
int main(){
	// the slops is always in decrecent form
	// DP( j ) = min ( i < j ) ( DP( i ) + b[ i ] a[ j ] )      b[ i ] > b[ j ] 
	int n , x;
	while( sc( n ) == 1 ){
		sz = 0;
		vi a , b;
		REP( i , n ) sc( x ) , a.pb( x );
		REP( i , n ) sc( x ) , b.pb( x );
		vector< ll > DP( n );
		add( b[ 0 ] , DP[ 0 ] );
		for( int j = 1 ; j < n ; ++j ){
			DP[ j ] = eval( a[ j ] );
			add( b[ j ] , DP[ j ] );
		}
		cout << DP[ n - 1 ] << '\n';
	}
}

/////////
// falta analizar lo del pos
struct line{
    ll m , b ;
    line(){}
    line( ll m , ll b ) : m( m ) , b( b ) {}
};

line H[N + 5];
int sz,pos;

bool check(line &l1, line &l2, line &l3){
    return (ld)(l3.b - l2.b) * (l1.m - l3.m) >= (ld)(l3.b - l1.b) * (l2.m - l3.m);
}

void add( ll a , ll b){
    line l( a , b );
    while( sz >= 2 && !check( H[sz - 2] , H[sz - 1] , l ) ) --sz;
    H[ sz ++ ] = l;
}

ll eval( int ind , ll x ){
    return H[ind].m * x + H[ind].b;
}

ll query( ll x ){
    while(pos + 1 < sz && eval(pos,x) > eval(pos + 1,x)) ++pos;
    return eval(pos,x);
}
//// knuth optimization
//12836_UVA
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

#define N 1000
#define INF (1<<29)

typedef long long ll;
typedef pair< int , int > pii;
typedef vector< int > vi;
typedef vector< vi > vvi;

int X[ N + 5 ];
int DP[ N + 5 ][ N + 5 ] , A[ N + 5 ][ N + 5 ];

int main(){
	int cases;
	sc( cases );
	REP( tc , cases ){
		int n;
		sc( n );
		vi v( n );
		REP( i , n ) sc( v[ i ] );
		vi LIS1( n , 1 ) , LIS2( n , 1 );
		REP( i , n ){
			REP( j , i ) 
				if( v[ i ] > v[ j ] ) LIS1[ i ] = max( LIS1[ i ] , 1 + LIS1[ j ] );
		}
		
		for( int i = n - 1 ; i >= 0 ; --i ){
			for( int j = n - 1 ; j > i ; --j )
				if( v[ i ] > v[ j ] ) LIS2[ i ] = max( LIS2[ i ] , 1 + LIS2[ j ] );
		}
		
		vi vec( n );
		REP( i , n ) vec[ i ] =  LIS1[ i ] + LIS2[ i ] - 1;
		X[ 0 ] = 0;
		REP( i , n ) X[ i + 1 ] = X[ i ] + vec[ i ];
		
		REP( i , n ) REP( j , i ) DP[ i ][ j ] = INF;
		for( int len = 1 ; len <= n ; len++ ){
			for( int L = 0 ; L + len - 1 < n ; L++ ){
				int R = L + len - 1;
				if( len <= 1 ){
					A[ L ][ R ] = L;
					DP[ L ][ R ] = 0;
					continue;
				}
				int lo = A[ L ][ R - 1 ] , hi = A[ L + 1 ][ R ];
				DP[ L ][ R ] = INF;
				for( int i = lo ; i <= hi ; i++ ){
					ll aux = DP[ L ][ i ] + DP[ i + 1 ][ R ] + (X[ R + 1 ] - X[ L ]);
					if( DP[ L ][ R ] > aux ){
						DP[ L ][ R ] = aux;
						A[ L ][ R ] = i;
					}
				}
			}
		}
		cout << "Case " << tc + 1 << ": " << DP[ 0 ][ n - 1 ] << '\n';
	}
}


##################################### BAIRSTOW METHOD AND POLYNOMIALS ###########################################

typedef long double ld;
typedef vector< ld > poly;

void solve_cuadratic( ld A , ld B , ld C , ld &re1 , ld &im1 , ld &re2 , ld &im2 )
{
	ld disc = B*B - 4*A*C;
	ld Disc = abs( disc );
	ld x = -B/( 2*A ) , y = sqrt( Disc )/( 2*A );
	if( Disc < EPS || disc < EPS ) re1 = x , im1 = y , re2 = x , im2 = -y;
	else re1 = x + y , im1 = 0 , re2 = x - y , im2 = 0;
}

void bairstow( poly a , int n , poly &B , ld re[] , ld im[] , int &sz )
{
	ld b[ n + 1 ] , f[ n + 1 ];
	ld c , d , g , h;
	b[ n ] = b[ n - 1 ] = 0;
	f[ n ] = f[ n - 1 ] = 0;
	ld u = -1 , v = -1;
	REP( t , 100 )
	{
		for( int i = n - 2 ; i >= 0 ; --i )
		{
			f[ i ] = b[ i + 2 ] - u*f[ i + 1 ] - v*f[ i + 2 ];
			b[ i ] = a[ i + 2 ] - u*b[ i + 1 ] - v*b[ i + 2 ];
		}
		c = a[ 1 ] - u*b[ 0 ] - v*b[ 1 ];
		d = a[ 0 ] - v*b[ 0 ];
		g = b[ 1 ] - u*f[ 0 ] - v*f[ 1 ];
		h = b[ 0 ] - v*f[ 0 ];
		u = u -( -h*c + g*d )/( v*g*g + h*(h-u*g) );
		v = v -( -g*v*c + ( g*u - h )*d)/( v*g*g + h*(h-u*g) );
	}
	B = poly( b , b + n - 1 );
	solve_cuadratic( 1 , u , v , re[ sz ] , im[ sz ] , re[ sz + 1 ] , im[ sz + 1 ] );
	sz += 2;
}

void solve( poly a , int n , ld re[] , ld im[] , int &sz , bool &solvable )
{
	if( n == 0 )
	{
		if( abs( a[ 0 ] ) <= EPS ) re[ sz ] = im[ sz ] = 0 , sz++ ;
		else solvable = 0;
		return;
	}
	if( abs( a[ n ] ) < EPS )
	{
		re[ sz ] = im[ sz ] = 0;
		sz++;
		a.pop_back();
		solve( a , n - 1 , re , im , sz , solvable );
		return ;
	}
	if( n == 1 )
	{
		re[ sz ] = -a[ 0 ]/a[ 1 ];
		im[ sz ] = 0;
		sz++;
		return;
	}
	if( n == 2 )
	{
		ld A = a[ 2 ] , B = a[ 1 ] , C = a[ 0 ];
		solve_cuadratic( A , B , C , re[ sz ] , im[ sz ] , re[ sz + 1 ] , im[ sz + 1 ] );
		sz += 2;
		return;
	}
	poly F;
	bairstow( a , n , F , re , im , sz );
	solve( F , n - 2 , re , im , sz , solvable );
}
poly fix( poly V )
{
	int nV = V.size();
	while( nV > 1 && abs( V[ nV - 1 ] ) < EPS )
		V.pop_back() , nV--;
	return V;
}
poly sum( poly A , poly B )
{
	int nA = A.size() , nB = B.size() , nV = max( nA , nB );
	poly V( nV );
	REP( i , nA ) V[ i ] += A[ i ];
	REP( i , nB ) V[ i ] += B[ i ];
	return fix( V );
}
poly prod( poly A , poly B )
{
	int nA = A.size() , nB = B.size() , nV = nA + nB;
	poly V( nV );
	REP( i , nA )REP( j , nB ) V[ i + j ] += A[ i ]*B[ j ];
	return fix( V );
}

poly Integral( poly P )
{
	int nP = P.size();
	poly V( nP + 1 );
	REP( i , nP ) V[ i + 1 ] = P[ i ]/( i + 1 );
	return V;
}
ld eval( poly &P , ld x )
{
	int nP = P.size();
	ld ans = 0;
	for( int i = nP - 1 ; i >= 0 ;--i )
		ans = ans*x + P[ i ];
	return ans;
}
##################################### BIG NUM ###########################################
// Desventaja en la division el divisor debe ser un numero peque\F1o <= num
// bignum
//#231 (Div. 2) B. Very Beautiful Number
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

#define pb push_back
#define all( v ) v.begin() , v.end()
#define SZ( v ) ((int)(v).size())

#define mp make_pair
#define fi first
#define se second

typedef pair< int , int > pii;
typedef long long ll;
typedef vector< int > vi;
typedef vector< ll > vll;

const int base = 1000*1000*1000;
void impr( vi &a ){
	printf ("%d", a.empty() ? 0 : a.back());
	for (int i=SZ(a)-2; i>=0; --i)
		printf ("%09d", a[i]);
}
int getLen( vi &a ){
	if( a.empty() ) return 1;
	int r = a.back();
	int ans = 0;
	while( r ){
		r /= 10;
		ans ++;
	}
	ans += 9 *( SZ( a ) - 1 );
	return ans;
}
vi get( string s ){
	vi a;
	for (int i=(int)s.length(); i>0; i-=9)
		if (i < 9)
			a.push_back (atoi (s.substr (0, i).c_str()));
		else
			a.push_back (atoi (s.substr (i-9, 9).c_str()));
	while (a.size() > 1 && a.back() == 0) a.pop_back();
	return a;
}
vi sum( vi a , vi b ){
	int carry = 0;
	for (size_t i=0; i<max(a.size(),b.size()) || carry; ++i) {
		if (i == a.size())
			a.push_back (0);
		a[i] += carry + (i < b.size() ? b[i] : 0);
		carry = a[i] >= base;
		if (carry)  a[i] -= base;
	}
	return a;
}
vi rest( vi a , vi b ){
	int carry = 0;
	for (size_t i=0; i<b.size() || carry; ++i) {
		a[i] -= carry + (i < b.size() ? b[i] : 0);
		carry = a[i] < 0;
		if (carry)  a[i] += base;
	}
	while (a.size() > 1 && a.back() == 0)
		a.pop_back();
	return a;
}
vi mult1( vi a , int b ){
	int carry = 0;
	for (size_t i=0; i<a.size() || carry; ++i) {
		if (i == a.size())
			a.push_back (0);
		long long cur = carry + a[i] * 1ll * b;
		a[i] = int (cur % base);
		carry = int (cur / base);
	}
	while (a.size() > 1 && a.back() == 0)
		a.pop_back();
	return a;
}
vi mult2( vi a , vi b ){
	vi c (a.size()+b.size());
	for (size_t i=0; i<a.size(); ++i)
		for (int j=0, carry=0; j<(int)b.size() || carry; ++j) {
			long long cur = c[i+j] + a[i] * 1ll * (j < (int)b.size() ? b[j] : 0) + carry;
			c[i+j] = int (cur % base);
			carry = int (cur / base);
		}
	while (c.size() > 1 && c.back() == 0)
		c.pop_back();
	return c;
}
vi divide( vi a , int b , int &car ){
	int carry = 0;
	for (int i=(int)a.size()-1; i>=0; --i) {
		long long cur = a[i] + carry * 1ll * base;
		a[i] = int (cur / b);
		carry = int (cur % b);
	}
	while (a.size() > 1 && a.back() == 0)
		a.pop_back();
	car = carry;
	return a;
}

int main(){
	int p , x;

	while( sc( p ) == 1 ){
		sc( x );
		if( p == 1 ){
            if( x == 1 ) printf( "%d\n" , p );
            else puts( "Impossible" );
            continue;
        }
        vi A = get( string( 1 , '1' ) + string( p - 2 , '0' ) );
        vi B = get( string( 1 , '1' ) + string( p - 1 , '0' ) );
        bool solve = 0;
		for( int d = 1 ; d <= 9 ; ++d ){
            vi P = get( string( 1 , '1' ) + string( p - 1 , '0' ) );
            P = rest( P , get( string( 1 , '0' + x ) ) );
            P = mult1( P , d );
            int mod;
            P = divide( P , 10 * x - 1 , mod );

            if( mod == 0 && getLen( P ) == p - 1 ){
                impr( P );
                printf( "%d\n" , d );
                solve = true;
                break;
            }
        }
        

        if( !solve ) puts( "Impossible" );
        
	}
}
//build constructor
##################################### TWO POINTERS ###########################################
	for( int lo = 0 , hi = -1 ; lo < n ; ++lo ){
		while( hi + 1 < n && condition( hi + 1 ) ){
			add( hi + 1 );
			hi ++;
		}
		erase( lo );
	}
##################################### GRUNDY NUMBERS ###########################################
int dp( , , , , , ){
	if( ..... ) return ...; // CASOS BASES
	if( used[ ][ ] ... [ ][ ] ) return memo[ ini ][ fin ];
	used[ ][ ] ... [ ][ ] = 1;
	int &dev = memo[ ][ ] ... [ ][ ] = 0;
	
	
	int sz = ........ + 1; /// TAMA\D1O DE LA CANTIDAD DE ESTADOS LLAMADOS
	vector< bool > SET( sz + 1 );
	for( .... ){
		REP( ....... ){
			if( .......... ){
				int sumG = dp( ... ) ^ dp( ..... ) ^ dp( ......... );
				if( sumG <= sz ) SET[ sumG ] = 1;
			}
		}
	}
	
	while( SET[ dev ] ) dev++;
	return dev;
}
////////////////////////////////////////////////// ENDEND ////////////////////////////////////////////////////////////////////////////


######################## Math ###################################
//SIMPSON'S FORMULES
http://en.wikipedia.org/wiki/Newton_cotes_rules
http://acm.tju.edu.cn/acm/showp3775.html

int n = 500; // n tiene que ser par
ld f( ld x ){ return sqrt( 1 + 1.0/x ) / x; }
int main(){
	int a , b ;
	while( cin >> a >> b ){
		ld dt = ( b - a )/(ld)n;
		ld I = 0 , pos = a;
		for( int i = 0 ; i < n ; i +=2 )
			I += dt * ( f( pos ) + 4*f( pos + dt ) + f( pos = pos + dt + dt ));
		printf( "%.2f\n" , double( I/3.0 ) );
	}
}


// Series conocidas 
// A000217 	 Triangular numbers: a(n) = C(n+1,2) = n(n+1)/2 = 0+1+2+...+n. 0, 1, 3, 6, 10, 15, 21, 28 ...( 0 , 0 + 1 , 0 + 1 + 2 ,...) 
// f* = (-1+sqrt( 8*x + 1 ))/2
// A000292 Tetrahedral (or triangular pyramidal) numbers: a(n) = C(n+2,3) = n*(n+1)*(n+2)/6. 0, 1, 4, 10, 20, 35, 56, 84, 120.... ( 0 , 0 + 1 , 0 + 1 + 3 , 0 + 1 + 3 + 6 , ..)
// A000010 Euler totient function phi(n): count numbers <= n and prime to n.  1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16, 6, 18, 8, 12, 10, 22, 8, 20, 12, 18, 12, 28, 8, 30, 16, 20, 16, 24, 12, 36, 18, 24, 16, 40, 12, 42, 20, 24, 22, 46, 16, 42, 20, 32, 24, 52, 18, 40, 24, 36, 28, 58, 16, 60, 30, 36, 32, 48, 20, 66, 32, 44
// A000108	Catalan numbers: C(n) = binomial(2n,n)/(n+1) = (2n)!/(n!(n+1)!). Also called Segner numbers.1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440, 9694845, 35357670, 129644790, 477638700, 1767263190, 6564120420, 24466267020, 91482563640, 343059613650, 1289904147324, 4861946401452, 18367353072152, 69533550916004, 263747951750360, 1002242216651368, 3814986502092304
// A000169  Number of labeled rooted trees with n nodes: n^(n-1).       1, 2, 9, 64, 625, 7776, 117649, 2097152, 43046721, 1000000000, 25937424601, 743008370688, 23298085122481, 793714773254144, 29192926025390625, 1152921504606846976, 48661191875666868481, 2185911559738696531968, 104127350297911241532841, 5242880000000000000000000
// A006717  Number of toroidal semi-queens on a (2n+1) X (2n+1) board.  1, 3, 15, 133, 2025, 37851, 1030367, 36362925, 1606008513, 87656896891, 5778121715415, 452794797220965, 41609568918940625
//Burnside's lemma
// http://en.wikipedia.org/wiki/Necklace_(combinatorics) /// DECORATE_CDC
//Propiedades de la mediana , n particular, m is a sample median if and only if m minimizes the arithmetic mean of the absolute deviations. E( |X - c| )
//Propiedades de la media , f it is required to use a single number as a "typical" value for a set of known numbers , then the arithmetic mean of the numbers does this best, in the sense of minimizing the sum of squared deviations from the typical value: the sum of  E( |X - c|^2 ) https://en.wikipedia.org/wiki/Arithmetic_mean
// REVISAR !! A140358 http://codeforces.com/problemset/problem/11/B
// http://en.wikipedia.org/wiki/Negative_bases 739_SPOJ
	while( cin >> n )
	{
		string ans;
		if( !n )ans = "0";
		while( n != 0 )
		{
			ll r = n%(-2);
			n = n/(-2);
			if( r < 0 )
				r += 2 , n ++;
			ans = string(1 , '0'+r) + ans ;
		}
		cout << ans << endl;
	} 
//Derangement In combinatorial mathematics, a derangement is a permutation of the elements of a set such that none of the elements appear in their original position.
http://en.wikipedia.org/wiki/Derangement
// DP[ n ] = ( n - 1 ) * ( DP[ n - 1 ] + DP[ n - 2 ] ) , DP[ 0 ] = 1 , DP[ 1 ] = 0;   11282_UVA
// Phisic MRU x(t) = v*t + x0


//// Teorema peruano del resto xD
// a^b%c = a^x%c
// if( b >= phi( c ) ) x = b%phi( c ) + phi( c );
//	else x = b
//
######################## Polinomial ###################################

vector<int> add(vector<int> &a, vector<int> &b){
    int n = a.size(),m = b.size(),sz = max(n,m);
    vector<int> c(sz,0);
	
    for(int i = 0;i<n;++i) c[i] += a[i];
    for(int i = 0;i<m;++i) c[i] += b[i];
	
    while(sz>1 && c[sz-1]==0){
        c.pop_back();
        --sz;
    }
	
    return c;
}

vector<int> multiply(vector<int> &a, vector<int> &b){
    int n = a.size(),m = b.size(),sz = n+m-1;
    vector<int> c(sz,0);
	
    for(int i = 0;i<n;++i)
        for(int j = 0;j<m;++j)
            c[i+j] += a[i]*b[j];
	
    while(sz>1 && c[sz-1]==0){
        c.pop_back();
        --sz;
    }
	
    return c;
}

bool is_root(vector<int> &P, int r){
    int n = P.size();
    long long y = 0;
	
    for(int i = 0;i<n;++i){
        if(abs(y-P[i])%r!=0) return false;
        y = (y-P[i])/r;
    }
	
    return y==0;
}


#-----------------------------------------------------------------#
####################### Newton's Method ####################################
	double xn = 1e100 , h = f(xn)/fd(xn);
	while( abs(h) > eps )
	{
		xn = xn - h ;
		h = f(xn)/fd(xn);    
	}
#-----------------------------------------------------------------#	
####################### Solve Cubic ####################################	
	double x = a , y = x*xn + b , z = y*xn + c;
	double disc = y*y - 4*x*z;
	
	double x0 = -y + sqrt( abs(disc) );
	double x1 = -y - sqrt( abs(disc) );
	x0 /= (2*x) , x1 /=(2*x);
#-----------------------------------------------------------------#	

############################# SORTING ##############################

//bool compara 
bool compara(int x,int y){
    return A[ x ] < A[ y ];
}
void doit(){
    REP( i , n ) ind[ i ] = i;
    sort( ind , ind + n , compara );
}
///SOrt rev 
**osea orden(int a, int b) te pregunta si a debe ir antes que b
a iria antes que b siempre y cuando a > b **
bool orden(int a, int b) { return a > b; }
sort(v, v + n, orden)
// SORTING STRING "case insenstive" no hay diferencia entre mayusculas y minusculas (strcasecmp(a.team_name.c_str(),b.team_name.c_str) ) < 0
// o  simplemente transformar a minusculas todos los caracteres y compararlos 
// transform puede ayudar con ello rapidamente ( transform( all(t) , t.begin() , ::tolower ); )

//comparar cadenas a lo mas de tama\F1o len
	t <= s ---- t - s <= 0
return strncmp( t , s , len ) <= 0;
	t < s ----- t - s < 0'
return strncmp( t , s , len ) < 0;

#-----------------------------------------------------------------#

######################## BITWISE###################################

	Para recorrer todos los subconjuntos(no vacios) del conjunto s hacemos
	int s;
	for( int mask = s ; mask != 0 ; mask = ( mask - 1 ) & s ){
		/**/
	}
	Para recorrer todos los subconjuntos(no vacios) de fixed que no tienen
	ningun elemento de pro se hace	
	int fixed = 15 , pro = 10;
	int allBits = ( 1 << N ) - 1;
	int s = ( fixed ^ pro ) & fixed;
	for( int mask = s ; mask != 0 ; mask = ( mask - 1 ) & s ){
		/**/
	}
Bit-Manipulation.
__builtin_ctz(unsigned int x)
Counts the number of trailing zero bits.
__builtin_clz(unsigned int x)
Counts the number of leading zero bits.
// Most significant bit (MSB)
int g( int mask )  return 1 << ( 31 - __builtin_clz( mask ) );
Note that both these functions are undefined if x is 0.
__builtin_popcount(unsigned int x)
Counts the total number of one bits in x.
for ll function ==> function + ll

/// Rango of C++ types

/********************************************************************
* Limits *
********************************************************************/
tipo 			|bits| 		m\EDnimo 	.. m\E1ximo
----------------+----+-----------------------------------
unsigned char 	| 8  | 			 0 	.. 127
char 			| 8  | 		  -128 	.. 127
unsigned char 	| 8  | 			 0 	.. 255
short 			| 16 | 	   -32.768 	.. 32.767
unsigned short 	| 16 | 			 0 	.. 65.535
int 			| 32 |  -2 \D7 10**9 	.. 2 \D7 10**9
unsigned int 	| 32 | 			 0 	.. 4 \D7 10**9
int64_t 		| 64 | -9 \D7 10**18 	.. 9 \D7 10**18
uint64_t 		| 64 | 			 0 ..  18 \D7 10**18
float 			| 32 | 10**38 with precision 6
double 			| 64 | 10**308 with precision 15
long double 	| 80 | 10**19.728 with precision 18

#-----------------------------------------------------------------#
Tablas y cotas
Primos
2 3 5 7 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107 109
113 127 131 137 139 149 151 157 163 167 173 179 181 191 193 197 199 211 223 227 229
233 239 241 251 257 263 269 271 277 281 283 293 307 311 313 317 331 337 347 349 353
359 367 373 379 383 389 397 401 409 419 421 431 433 439 443 449 457 461 463 467 479
487 491 499 503 509 521 523 541 547 557 563 569 571 577 587 593 599 601 607 613 617
619 631 641 643 647 653 659 661 673 677 683 691 701 709 719 727 733 739 743 751 757
761 769 773 787 797 809 811 821 823 827 829 839 853 857 859 863 877 881 883 887 907
911 919 929 937 941 947 953 967 971 977 983 991 997 1009 1013 1019 1021 1031 1033
1039 1049 1051 1061 1063 1069 1087 1091 1093 1097 1103 1109 1117 1123 1129 1151
1153 1163 1171 1181 1187 1193 1201 1213 1217 1223 1229 1231 1237 1249 1259 1277
1279 1283 1289 1291 1297 1301 1303 1307 1319 1321 1327 1361 1367 1373 1381 1399
1409 1423 1427 1429 1433 1439 1447 1451 1453 1459 1471 1481 1483 1487 1489 1493
1499 1511 1523 1531 1543 1549 1553 1559 1567 1571 1579 1583 1597 1601 1607 1609
1613 1619 1621 1627 1637 1657 1663 1667 1669 1693 1697 1699 1709 1721 1723 1733
1741 1747 1753 1759 1777 1783 1787 1789 1801 1811 1823 1831 1847 1861 1867 1871
1873 1877 1879 1889 1901 1907 1913 1931 1933 1949 1951 1973 1979 1987 1993 1997
1999 2003 2011 2017 2027 2029 2039 2053 2063 2069 2081
Primos cercanos a 10^n
9941 9949 9967 9973 10007 10009 10037 10039 10061 10067 10069 10079
99961 99971 99989 99991 100003 100019 100043 100049 100057 100069
999959 999961 999979 999983 1000003 1000033 1000037 1000039
9999943 9999971 9999973 9999991 10000019 10000079 10000103 10000121
99999941 99999959 99999971 99999989 100000007 100000037 100000039 100000049
999999893 999999929 999999937 1000000007 1000000009 1000000021 1000000033
Cantidad de primos menores que 10^n
(10^1) = 4 ; (10^2) = 25 ; (10^3) = 168 ; (10^4) = 1229 ; (10^5) = 9592
(10^6) = 78.498 ; (10^7) = 664.579 ; (10^8) = 5.761.455 ; (10^9) = 50.847.534
(10^10) = 455.052,511 ; (10^11) = 4.118.054.813 ; (10^12) = 37.607.912.018
Divisores
Cantidad de divisores (1) para algunos n
0(60) = 12 ; 0(120) = 16 ; 0(180) = 18 ; 0(240) = 20 ; 0(360) = 24
0(720) = 30 ; 0(840) = 32 ; 0(1260) = 36 ; 0(1680) = 40 ; 0(10080) = 72
0(15120) = 80 ; 0(50400) = 108 ; 0(83160) = 128 ; 0(110880) = 144
0(498960) = 200 ; 0(554400) = 216 ; 0(1081080) = 256 ; 0(1441440) = 288
0(4324320) = 384 ; 0(8648640) = 448

	// 100 G = 60 ===>  12
	// 1000 G = 840 ===> 32
	// 10000 G = 7560 ===> 64
	// 100000 G = 83160 ===> 128
	// 1000000 G = 720720 ===> 240
	// 10000000 G = 8648640 ===> 448
	// 100000000 G = 73513440 ==> 768
	//http://wwwhomes.uni-bielefeld.de/achim/highly.txt
	   no       number      divisors    2 3 5 71113171923293137414347535961677173
	   	176 9.200528e+019       245760   7 4 3 2 1 1 1 1 1 1 1 1 1
		166 9.200528e+018       161280   6 4 2 2 1 1 1 1 1 1 1 1 1
   		155 8.976125e+017       103680   8 4 2 2 1 1 1 1 1 1 1 1		
   		145 7.480104e+016        64512   6 3 2 2 1 1 1 1 1 1 1 1   		
   		134 8.086599e+015        41472   8 3 2 2 1 1 1 1 1 1 1   
   		124 8.664213e+014        26880   6 4 2 1 1 1 1 1 1 1 1		   		
   		116 9.782176e+013        17280   5 4 2 2 1 1 1 1 1 1   		
   		105 9.316358e+012        10752   6 3 2 1 1 1 1 1 1 1   		
		94  963761198400         6720   6 4 2 1 1 1 1 1 1   		
		85   97772875200         4032   6 3 2 2 1 1 1 1
    	75    6983776800         2304   5 3 2 1 1 1 1 1		
    	65     735134400         1344   6 3 2 1 1 1 1
    	55      73513440          768   5 3 1 1 1 1 1
		46       8648640          448   6 3 1 1 1 1
    	37        720720          240   4 2 1 1 1 1	
		28         83160          128   3 3 1 1 1	
    	19          7560           64   3 3 1 1		
    	14           840           32   3 1 1 1	
    	8            60           12   2 1 1			
    	3             6            4   1 1   
Suma de divisores (1) para algunos n
1(96) = 252 ; 1(108) = 280 ; 1(120) = 360 ; 1(144) = 403 ; 1(168) = 480
1(960) = 3048 ; 1(1008) = 3224 ; 1(1080) = 3600 ; 1(1200) = 3844
1(4620) = 16128 ; 1(4680) = 16380 ; 1(5040) = 19344 ; 1(5760) = 19890
1(8820) = 31122 ; 1(9240) = 34560 ; 1(10080) = 39312 ; 1(10920) = 40320
1(32760) = 131040 ; 1(35280) = 137826 ; 1(36960) = 145152 ; 1(37800) = 148800
1(60480) = 243840 ; 1(64680) = 246240 ; 1(65520) = 270816 ; 1(70560) = 280098
1(95760) = 386880 ; 1(98280) = 403200 ; 1(100800) = 409448
1(491400) = 2083200 ; 1(498960) = 2160576 ; 1(514080) = 2177280
1(982800) = 4305280 ; 1(997920) = 4390848 ; 1(1048320) = 4464096
1(4979520) = 22189440 ; 1(4989600) = 22686048 ; 1(5045040) = 23154768
1(9896040) = 44323200 ; 1(9959040) = 44553600 ; 1(9979200) = 45732192
5.3.3. Factoriales
0! = 1 			11! = 39.916.800
1! = 1 			12! = 479.001.600 (int)
2! = 2 			13! = 6.227.020.800
3! = 6 			14! = 87.178.291.200
4! = 24 		15! = 1.307.674.368.000
5! = 120 		16! = 20.922.789.888.000
6! = 720 		17! = 355.687.428.096.000
7! = 5.040 		18! = 6.402.373.705.728.000
8! = 40.320 	19! = 121.645.100.408.832.000
9! = 362.880 	20! = 2.432.902.008.176.640.000 (tint)
10! = 3.628.800 21! = 51.090.942.171.709.400.000
max signed tint = 9.223.372.036.854.775.807
max unsigned tint = 18.446.744.073.709.551.615

//Pascal triangle elements:
C(33, 16) = 1.166.803.110 [int limit]
C(34, 17) = 2.333.606.220 [unsigned int limit]
C(66, 33) = 7.219.428.434.016.265.740 [int64_t limit]
C(67, 33) = 14.226.520.737.620.288.370 [uint64_t limit]
//Fatorial
12! = 479.001.600 [(unsigned) int limit]
20! = 2.432.902.008.176.640.000 [(unsigned) int64_t limit ]
Some 'medium' primes:
8837 8839 8849 8861 8863 8867 8887 8893 8923 8929 8933
8941 8951 8963 8969 8971 8999 9001 9007 9011 9013 9029
9041 9043 9049 9059 9067 9091 9103 9109 9127 9133 9137
Cousins: (229,233),(277,281) Twins: (311,313),(347,349),(419,421)
Some not so small primes:
80963,263911,263927,567899,713681,413683,80963
37625701, 236422117, 9589487273, 9589487329, 694622169483311

#-----------------------------------------------------------------#

########################## ROYTRACKING #################################
// EL FAMOSO ROYTRACKING
// CODE DEL  ACM_4887
// FALTA ENTENDERLO Y SABER USAR Y APLICAR
#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;

string g(int n)
{
	stringstream os;
	os<<n;
	return os.str();
}

string f(int rank)
{
	if(rank == 1) return "1st";
	if(rank == 2) return "2nd";
	if(rank == 3) return "3rd";
	return g(rank) + "th";
}
 
int main()
{
	int n, m, caso = 0;
	while(cin>>n>>m)
	{
		if(!n && !m) break;
		
		caso++;
		if(caso != 1) cout<<endl;
		
		map <string, int> ID;
		
		vector <string> team(n);
		for(int i=0; i<n; i++)
		{
			cin>>team[i];
			ID[team[i]] = i;
		}
		
		vector <int> score(n, 0);
		vector <int> v1, v2;
		
		for(int i=0; i<m; i++)
		{
			string s1, s2;
			cin>>s1>>s2>>s2;
			s2 = s2.substr(0, (int)s2.size() - 1);
			
			int n1, n2;
			cin>>n1>>n2;
			
			if(n1 != -1 && n2 != -1)
			{
				if(n1 > n2) score[ID[s1]] += 3;
				else if(n1 < n2) score[ID[s2]] += 3;
				else
				{
					score[ID[s1]] += 1;
					score[ID[s2]] += 1;
				}
			}
			else
			{
				v1.push_back(ID[s1]);
				v2.push_back(ID[s2]);
			}
		}
		
		int minRank[n], maxRank[n];
		for(int i=0; i<n; i++)
		{
			minRank[i] = 1<<30;
			maxRank[i] = 0;
		}
		
		int k = v1.size();
		
		vector <int> mask(k, 0);
		while(1)
		{
			vector <int> nscore = score;
			for(int i=0; i<k; i++)
			{
				if(mask[i] == 0) nscore[v1[i]] += 3;
				else if(mask[i] == 1) nscore[v2[i]] += 3;
				else
				{
					nscore[v1[i]] += 1;
					nscore[v2[i]] += 1;
				}
			}
			
			vector < pair <int, int> > v;
			for(int i=0; i<nscore.size(); i++)
				v.push_back(make_pair(nscore[i], i));
			
			sort(v.rbegin(), v.rend());
			
			int p = 1, last = v[0].first;
			for(int i=0; i<v.size(); i++)
			{
				if(v[i].first < last) p = i + 1;
				last = v[i].first;
				
				minRank[v[i].second] = min(minRank[v[i].second], p);
				maxRank[v[i].second] = max(maxRank[v[i].second], p);
			}
			
			bool sigue = false;
			
			for(int i=0; i<k; i++)
			{
				if(mask[i] < 2)
				{
					mask[i]++;
					sigue = true;
					break;
				}
				else mask[i] = 0;
			}
			
			if(!sigue) break;
		}
		
		for(int i=0; i<n; i++)
			cout<<"Team "<<team[i]<<" can finish as high as "<<f(minRank[i])<<" place and as low as "<<f(maxRank[i])<<" place."<<endl;
	}
	
	return 0;
}
#-----------------------------------------------------------------#
########################## Polinomial #################################
// Parametrizar y comparar con la polinomial de MarioYC
#define MAXN 12
double P[MAXN];
double nP[2*MAXN];
double sP[2*MAXN];
double xlow,xhigh,inc;
int n;
double Peval(double x){
     double y=0;
     for( int i = 2*n+1 ; i >= 0 ; --i)
          y = y*x + sP[i];
     
     return y;
}
double f(double x){
     return pi*(Peval(x)-Peval(xlow));     
}
void square(){
     for(int i = 0 ; i<2*MAXN ;++i)
          nP[i]=0;
     
     for( int i = 0 ; i <= n ; ++i)
          for( int j = 0 ; j <= n ; ++j)
               nP[i+j]+=P[i]*P[j];                        
}
void Integral(){
     sP[0]=1;
     for(int i = 1 ; i <= 2*n+1 ; ++i)
          sP[i] = nP[i-1]/i;     
}
#-----------------------------------------------------------------#

#-----------------------------------------------------------------#
########################## FECHAS (MyDate) #################################
//C:\luis\HackerRank\ProyectEuler\euler019.cpp
//Project Euler #19: Counting Sundays

string D[] = { "" , "Lunes" , "Martes" , "Miercoles" , "Jueves" , "Viernes" , "Sabado" , "Domingo" , ""};
string M[] = { "" , "Enero" , "Febrero" , "Marzo" , "Abril" , "Mayo" , "Junio" , "Julio" , "Agosto" , "Setiembre" , "Octubre" , "Noviembre" , "Diciembre" , "" };
ll Tot[] = { 0 , 31 , 28 , 31 , 30 , 31 , 30 , 31 , 31 , 30 , 31 , 30 , 31 , 0 };
//1 Jan 1900 was a Monday.
///En days Lunes  == 1 , Martes == 2 ..etc
// modulus lunes = 1 , martes = 2 ..... saturday = 6 sunday = 0
// property the days of the week repeats every 400 years this because taking segments of years [ 0 400 > [ 400 800 ]...
// number of day(including leap years) = (365*400 + 400/4-400/100+400/400) mod 7 = 0
bool is_leap_year( ll year ){ return  ( ( ( year % 4 == 0 ) && ( year % 100 != 0) ) || ( year % 400 == 0) ); }
ll tot_day_for_month( ll mes , ll year ){ 
	return ( ( mes == 2 && is_leap_year( year ) ) ? 29 : Tot[ mes ] );
}
ll countBis( ll x ){ 
	return (x / 4LL) - (x / 100LL) + (x / 400LL); 
}
ll dayCount( ll day, ll month , ll year ) {
	ll days = 365LL * ( year - 1LL );
	days += countBis( year - 1LL );
	for( int m = 1 ; m < month ; m++ ) days += tot_day_for_month( m , year );
	return days + day ;
}
bool valid( ll d , ll m , ll y ){
	if( m >= 1 && m <= 12 ){
		if( d >= 1 && d <= tot_day_for_month( m , y ) ) return 1;
	}
	return 0;
}
void prevDate( ll d , ll m , ll y , ll &dd , ll &mm , ll &yy ){
	dd = d - 1 , mm = m , yy = y;
	if( valid( dd , mm , yy ) ) return;
	mm = m - 1 , yy = y;
	dd = tot_day_for_month( mm , yy );
	if( valid( dd , mm , yy ) ) return;
	dd = 31 , mm = 12 , yy = y - 1;
}

ll cnt = 0;
ll F[ 400 + 5 ][ 12 + 5 ];
ll get( ll d , ll m , ll y ){
	return (y / 400) * cnt + F[ y % 400 ][ m ];
}
int main(){
	ll X = dayCount( 1 , 1 , 2000 );
	for( int y = 2000 ; y < 2400 ; ++y ){
		for( int m = 1 ; m <= 12 ; ++m ){
			if( X % 7 == 0 ) cnt ++;
			F[ y - 2000 ][ m ] = cnt;
			X += tot_day_for_month( m , y );
		}
	}
	int cases;
	sc( cases );
	REP( tc , cases ){
		ll y1 , m1 , d1;
		ll y2 , m2 , d2;
		scanf( "%lld%lld%lld" , &y1 , &m1 , &d1 );
		scanf( "%lld%lld%lld" , &y2 , &m2 , &d2 );
		prevDate( d1 , m1 , y1 , d1 , m1 , y1 );
		ll A = get( d1 , m1 , y1 ) , B = get( d2 , m2 , y2 );
		printf( "%lld\n" , B - A );
	}
}

#-----------------------------------------------------------------#

########################## FECHAS (MyDate) JAVA #################################

import java.util.Calendar;

public class MyDate {

    public int day;
    public int month;
    public int year;
    public boolean leap_year;
    public String day_name;
    public String month_name;

    public static int Tot[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 , 0 };
    public static String D[] = { "","Lunes" , "Martes" , "Miercoles" , "Jueves" , "Viernes" , "Sabado" , "Domingo" , "" };
    public static String M[] = { "","Enero" , "Febrero" , "Marzo" , "Abril" , "Mayo" , "Junio" , "Julio" , "Agosto" , "Setiembre" ,"Octubre" , "Noviembre" , "Diciembre" , "" };

    public MyDate() {
        this.day = 1;
        this.month = 1;
        this.year = 1900;
        this.leap_year = is_leap_year(year);
        this.day_name = D[1];
        this.month_name = M[1];
    }
    public MyDate(int _day, int _month, int _year) {
        this.day = _day;
        this.month = _month;
        this.year = _year;
        if( validate() )
        {
            this.leap_year = is_leap_year( year);
            this.day_name = getDay_name( day , month, year);
            this.month_name = getMonth_name( month );
        }
    }
    public void actualMyDate() {
        Calendar calendar = Calendar.getInstance();
        int _year = calendar.get(Calendar.YEAR);
        int _month = calendar.get(Calendar.MONTH);
        int _day = calendar.get(Calendar.DATE);
        this.day = _day;
        this.month = _month + 1;
        this.year = _year;
        if( validate() )
        {
            this.leap_year = is_leap_year( year);
            this.day_name = getDay_name( day , month, year);
            this.month_name = getMonth_name( month );
        }
    }

    public String Weather_station() {
        /*
        Las estaciones en Per\FA (hemisferio sur) :
        Verano: del 22 de diciembre al 21 de marzo.
        Oto\F1o: del 22 de marzo al 21 de junio.

        Invierno:del 22 de junio al 22 de septiembre.
        Primavera: del 23 de septiembre al 21 de diciembre.
         */
        int []d = { 0 , 1, 21 , 22 , 21 , 22 , 22 , 23 , 21 , 22 , 31 , 0 };
        int []m = { 0 , 1 , 3 , 3  , 6  , 6  , 9  , 9  , 12 , 12 , 12 , 0 };
        String station[] = { "Verano" , "Oto\F1o" , "Invierno" , "Primavera" , "Verano" };
        MyDate []f = new MyDate[11];
        for( int i = 0 ; i < 11 ; ++i )
            f[i] = new MyDate(d[i], m[i], year);
        MyDate F = new MyDate(day, month, year);
        for( int i = 0 ; i < 5 ; ++i )
            if( f[1+2*i].compareDate(F)&&F.compareDate(f[2+2*i]) )
                return station[i];
        return "";
    }
    public int ageInWeeks(MyDate f1, MyDate f2) {
        return (f2.dayCount() - f1.prevDate().dayCount()) / 7;
    }
    // r == 1 Lunes , r == 2 Martes .....
    public int dayOfWeekCount(int r) {
        int k = dayCount();
        return k/7 + ((k%7)>= r?1:0);
    }
    /// operator <=
    public boolean compareDate(MyDate f1) {
        int p = year * 10000 + month * 100 + day;
        int q = f1.year * 10000 + f1.month * 100 + f1.day;
        return p <= q;
    }
    public MyDate nextDate() {
        MyDate f2 = new MyDate(day + 1, month, year);
        if (f2.validate()) return f2;
        f2 = new MyDate(1, month + 1, year);
        if (f2.validate()) return f2;
        f2 = new MyDate(1, 1, year + 1);
        return f2;
    }
    public MyDate prevDate() {
        MyDate f2 = new MyDate(day - 1, month, year);
        if (f2.validate()) return f2;
        f2 = new MyDate(day, month - 1, year);
        f2 = new MyDate(tot_day_for_month(f2.month, f2.year), month - 1, year);
        if (f2.validate()) return f2;
        f2 = new MyDate(31, 12, year - 1);
        return f2;
    }
    public void showDate() {
        System.out.println(day + "/" + month + "/" + year + " " + day_name + " " + month_name);
    }
    public static void showDates(MyDate f1, MyDate f2) {
        while (f1.compareDate(f2)) {
            f1.showDate();
            f1 = f1.nextDate();
        }
    }
    public static int dayCount(int _day, int _month, int _year ) {
        int days = 0;
	for ( int a = 1900; a < _year ; a++) days += ( is_leap_year(a) ? 366 : 365 );
	for ( int m = 1; m < _month; m++) days += tot_day_for_month( m , _year ) ;
	return days + _day ;
    }
    public int dayCount() {
        return dayCount(day, month, year);
    }
    public static String getDay_name(int _day, int _month, int _year ) {
        int r = dayCount(_day, _month, _year);
        r--;r%=7;r++;
        return D[r];
    }
    public boolean validate() {
        return (year >= 1900) && (month >= 1 && month <= 12) && (day >= 1 && day <= tot_day_for_month(month, year));
    }
    public static String getMonth_name(int _month) {
        return M[ _month ];
    }
    public static int tot_day_for_month(int _month, int _year) {
        return ((_month == 2 && is_leap_year(_year)) ? 29 : Tot[_month]);
    }
    public static boolean is_leap_year( int _year ) {
        return  ( ( (_year % 4 == 0 ) && (_year % 100 != 0) ) || ( _year % 400 == 0) );
    }
    public boolean equals( MyDate obj ){
        return obj.day == day && obj.month == month && obj.year == year;
    }
    public static void main(String[] args) {

    }
}


################# Fractions ##########################################

//fractions JAVA 100210B_GYM

public class Main  {
    public static BigInteger ONE = BigInteger.ONE;
    public static BigInteger ZERO = BigInteger.ZERO;
    public static BigInteger TWO = BigInteger.valueOf(2);
    static class frac{
        BigInteger num , den;
        frac(){ num = den = ZERO;}
        public frac( BigInteger a , BigInteger b ){
                if( b.equals( ZERO ) ) {
                    num = ONE; 
                    den = ZERO;
                } else if( a.equals( ZERO ) ){ 
                    num = ZERO; 
                    den = ONE;
                }else{
                        if( b.compareTo(ZERO) == -1 ){
                            a = a.negate(); 
                            b = b.negate();
                        }
                        BigInteger g = (a.abs()).gcd(b.abs());
                        num = a.divide(g) ; 
                        den = b.divide(g);
                }
        }
    };
    public static boolean Less( frac a , frac b ){
            return ( a.num.multiply( b.den ) ).compareTo( b.num.multiply( a.den ) ) == -1;
    }
    public static boolean Higher( frac a , frac b ){
            return ( a.num.multiply( b.den ) ).compareTo( b.num.multiply( a.den ) ) == 1;
    }
    boolean equals ( frac A , frac B ){ return A.num.equals(B.num) && A.den.equals(B.den);}
    public static frac sum( frac A , frac B ){ return new frac( A.num .multiply( B.den ).add( B.num .multiply( A.den ) ) , A.den.multiply( B.den ) ) ;}
    public static frac resta( frac A , frac B ){ return new frac( A.num .multiply( B.den ).subtract( B.num .multiply( A.den ) ) , A.den.multiply( B.den ) ) ;}

    public static frac mult( frac A , frac B ){ return new frac( A.num.multiply( B.num ) , A.den.multiply( B.den ) ) ;}
    public static frac div( frac A , frac B ){ return new frac( A.num.multiply( B.den ) , A.den.multiply( B.num ) ) ;}
    public static frac abs( frac A ){ return new frac( A.num.abs() , A.den.abs() );}

    public static void invert( frac A[][] , frac B[][] , int n ){
		for( int i = 0 ; i < n ; ++i ){
			int jmax = i ;
			for( int j = i + 1 ; j < n ; ++j )
				if( Higher( abs( A[ j ][ i ] ) , abs( A[ jmax ][ i ] ) ) ) jmax = j;
			for( int j = 0 ; j < n ; ++j ){
	                    frac temp;
	                    temp = A[ i ][ j ];
	                    A[ i ][ j ] = A[ jmax ][ j ];
	                    A[ jmax ][ j ] = temp;
	                    
	                    temp = B[ i ][ j ];
	                    B[ i ][ j ] = B[ jmax ][ j ];
	                    B[ jmax ][ j ] = temp;
	                }
	
			frac tmp = A[ i ][ i ];
			for( int j = 0 ; j < n ; ++j ){
	                    A[ i ][ j ] = div( A[ i ][ j ] , tmp );
	                    B[ i ][ j ] = div( B[ i ][ j ] , tmp );
	                }
			for( int j = 0 ; j < n ; ++j ){
				if( i == j )continue;
				tmp = A[ j ][ i ];
				for( int k = 0 ; k < n ; ++k ){
	                A[ j ][ k ] = resta( A[ j ][ k ] , mult( A[ i ][ k ] , tmp ) );
	                B[ j ][ k ] = resta( B[ j ][ k ] , mult( B[ i ][ k ] , tmp ) );
	            }
			}
		}
    }
    public static String input = "divide.in";
    public static String output = "divide.out";
    
    public static void main(String[] args) throws FileNotFoundException {
        Scanner cin = new Scanner( new File(input) );
        int n = cin.nextInt();
        cin.close();
        
        PrintWriter wt = new PrintWriter(output);
        frac A[][] = new frac[ n + 1 ][ n + 1 ];
        for( int i = 0 ; i <= n ; ++i )
            for( int j = 0 ; j <= n ; ++j ) A[ i ][ j ] = new frac( ONE , ONE );
        frac B[][] = new frac[ n + 1 ][ n + 1 ];
        for( int i = 0 ; i <= n ; ++i )
            for( int j = 0 ; j <= n ; ++j ) B[ i ][ j ] = new frac( ONE , ONE );
        for( int i = 0 ; i <= n ; ++i ) A[ i ][ n ] = new frac( ONE , ONE );
        for( int i = 0 ; i <= n ; ++i )
            for( int j = n - 1 ; j >= 0 ; --j )
                A[ i ][ j ] = mult( new frac( BigInteger.valueOf(i) , ONE ) , A[ i ][ j + 1 ] );
        B[ 0 ][ 0 ] = new frac( ONE , ONE );
        for( int i = 1 ; i <= n ; ++i ) B[ i ][ 0 ] = mult( new frac( TWO , ONE ) , B[ i - 1 ][ 0 ] );
        invert( A , B , n + 1 );
        wt.println( n );
        for( int i = 0 ; i <= n ; ++i ){
            wt.print( B[ i ][ 0 ].num + "/" + B[ i ][ 0 ].den );
            if( i == n ) wt.println("");
            else wt.print(" ");
        }
        wt.close();
    }
}

// fractions c++
//WARNING si vas a usar fracciones , esta estructura es un poco lenta comparada con double o long double
// solo usar si es necesario evitar errores de precision
struct frac{
	ll num , den;
	frac(){ num = den = 0;}
	frac( ll a , ll b ){
		if( b == 0 ) num = 1 , den = 0;
		else if( a == 0 ) num = 0 , den = 1;
		else{
			if( b < 0 ) a = -a , b = -b;
			ll g = __gcd( abs( a ) , abs( b ) );
			num = a/g , den = b/g;
		}
	}
};
bool operator <( const frac &a , const frac &b ){
	return a.num * b.den < b.num * a.den;
}
bool operator >( const frac &a , const frac &b ){
	return a.num * b.den > b.num * a.den;
}
bool operator == ( const frac &A , const frac &B ){ return A.num == B.num && A.den == B.den;}
bool operator != ( const frac &A , const frac &B ){ return !(A == B);}
frac operator +( const frac &A , const frac &B ){ return frac( A.num * B.den + B.num * A.den , A.den * B.den ) ;}
frac operator -( const frac &A , const frac &B ){ return frac( A.num * B.den - B.num * A.den , A.den * B.den ) ;}

frac operator *( const frac &A , const frac &B ){ return frac( A.num * B.num , A.den * B.den ) ;}
frac operator /( const frac &A , const frac &B ){ return frac( A.num * B.den , A.den * B.num ) ;}
frac abs( frac &A ){ return frac( abs( A.num ) , abs( A.den ) );}


################# OBSERVACIONES STL ##########################################

/////////////////////////////////////////////////////////////////////////////////////////////
	std::accumulate	
	#include <numeric>
	Descripcion : Devuelve la suma en un intervalo [a,b>
	sintaxis : accumulate(numbers,numbers+3,init)
	Ejemplo : para enteros accumulate( a , a + n , 0 );
			para string strings , funciona como concatenacion  
			int main()
			{
				vector< string > v ;
				string s = accumulate( v.begin() , v.end() , string("") );
			}
/////////////////////////////////////////////////////////////////////////////////////////////
	std::merge
	#include <algorithm>	
	Merge sorted ranges
	Combines the elements in the sorted ranges [first1,last1) and [first2,last2), into a new range beginning 
	at result with all its elements sorted.
	The objects in the range between result and the returned value are modified.
		std::merge (first,first+5,second,second+5,v.begin());
/////////////////////////////////////////////////////////////////////////////////////////////
	std::string::find
	#include <string>	
	std::string::npos 	// NO MATCH (-1)
	Find content in string
  	string		unsigned found = str.find(str2);		 
	c-string	str.find("needles are small",found+1,6);
	sbuffer		tr.find("haystack");
	character	str.find('.');
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::string::rfind
	#include <string>	
	Find last occurrence of content in string
	string		str.rfind(key)
/////////////////////////////////////////////////////////////////////////////////////////////		
	std::find
	#include <algorithm> 
	Find value in range
	Returns an iterator to the first element in the range [first,last) that compares equal to val. 
	If no such element is found, the function returns last.
	it = find (myvector.begin(), myvector.end(), 30);
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::string::substr
	#include <string>	
	Generate substring
	str.substr (pos,tam); Create Substr from pos of size tam
	str.substr (pos);  Create sufix from pos
/////////////////////////////////////////////////////////////////////////////////////////////
	std::pair::pair
	#include<utility>
	pair <string,double> product1;                     // default constructor
	pair <string,double> product2 ("tomatoes",2.30);   // value init
	pair <string,double> product3 (product2);          // copy constructor
/////////////////////////////////////////////////////////////////////////////////////////////	
	Object :: iterator it;
	Example:
		set<int>::iterator it;
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::set::set
	#include<set>
	The elements are sorted according to the comparison object	.
	Unique element values: no two elements in the set can compare equal to each other. 	
	set<int> first;                           // empty set of ints
	int myints[]= {10,20,30,40,50};
	set<int> second (myints,myints+5);        // range
	set<int> third (second);                  // a copy of second
	set<int> fourth (second.begin(), second.end());  // iterator ctor.	
/////////////////////////////////////////////////////////////////////////////////////////////
	std::set::insert
	#include<set>
	Extends the container by inserting new elements, effectively increasing the container 
	size by the number of elements inserted.
    	int v[] = { 1 , 5 , 4 , 2 , 3};
		set< int >S( v , v + 5 );
		set< int >T( S );
		int x  = 80 ;
		T.insert( x );
		set< int > :: iterator it = T.begin();
		it++ , it++;
		T.insert( it ,x + 1 );
		set< int >U;
		U.insert( all(T) );
		FOR( o , U )cout << *o << endl;	
	Complexitys
	For the first version ( insert(x) ), logarithmic.
	For the second version ( insert(position,x) ), logarithmic in general, 
	but amortized constant if x is inserted right after the element pointed by position.
	For the third version ( insert (first,last) ), Nlog(size+N) in general 
	(where N is the distance between first and last, and size the size of the container 
	before the insertion), but linear if the elements between first and last are already }
	sorted according to the same ordering criterion used by the container.
/////////////////////////////////////////////////////////////////////////////////////////////
	std::set::find
	#include<set>
	Searches the container for an element equivalent to val and returns an iterator to it if found, 
	otherwise it returns an iterator to set::end.
		set< int > S;
		set< int > :: iterator it;
		it = S.find(20);  
	Complexity
	Logarithmic in size.
/////////////////////////////////////////////////////////////////////////////////////////////
	std::set::count
	#include<set>
	Searches the container for elements equivalent to val and returns the number of matches.
	Because all elements in a set container are unique, the function can only return 1 
	(if the element is found) or zero (otherwise).
		if (myset.count(i)!=0)
			cout << " is an element of myset.\n";
	    else
			cout << " is not an element of myset.\n";
	Complexity
	Logarithmic in size.
///////////////////////////////////////////////////////////////////////////////////////////// 
	std::distance
	#include <iterator>
	std::list<int>::iterator first = mylist.begin();
	std::list<int>::iterator last = mylist.end();
	std::cout << "The distance is: " << distance(first,last) << '\n';
	Return distance between iterators
	Calculates the number of elements between first and last.
///////////////////////////////////////////////////////////////////////////////////////////// 
	std::advance
	std::advance (it,5);
	std::cout << "The sixth element in mylist is: " << *it << '\n';
	Advance iterator
	Advances the iterator it by n element positions.
///////////////////////////////////////////////////////////////////////////////////////////// 
	std::vector::resize
	#include <vector>
	Resizes the container so that it contains n elements.
	If n is smaller than the current container size, the content is reduced to its first n elements, removing those beyond 
	(and destroying them).
	If n is greater than the current container size, the content is expanded by inserting at the end as many elements 
	as needed to reach a size of n. If val is specified, the new elements are initialized as copies of val, otherwise, 
	they are value-initialized.
	If n is also greater than the current container capacity, an automatic reallocation of the allocated storage space takes place.
	Notice that this function changes the actual content of the container by inserting or erasing elements from it.
		vector<int> v;
		for (int i=1;i<10;i++) v.push_back(i);
		v.resize(5);
	  	v.resize(8,100);
		v.resize(12);
	myvector contains: 1 2 3 4 5 100 100 100 0 0 0 0
	Complexity
	Linear on the number of elements inserted/erased (constructions/destructions).
///////////////////////////////////////////////////////////////////////////////////////////// 
	std::string::resize
	#include <vector>
	Similar to vector
		str.resize (sz+2,'+');
		str.resize (14);
	Complexity
	Linear on the number of elements inserted/erased (constructions/destructions).
/////////////////////////////////////////////////////////////////////////////////////////////
	std::next_permutation
	#include<algorithm>
	Rearranges the elements in the range [first,last) into the next lexicographically greater permutation.
	If the function can determine the next higher permutation, it rearranges the elements as such and returns true. 
	If that was not possible (because it is already at the largest possible permutation), it rearranges the
	elements according to the first permutation (sorted in ascending order) and returns false.
		do {
			cout << myints[0] << " " << myints[1] << " " << myints[2] << endl;
		} while ( next_permutation (myints,myints+3) );
	Complexity
	Up to linear in half the distance between first and last (in terms of actual swaps). (n!)
/////////////////////////////////////////////////////////////////////////////////////////////
	#include<cctype>
	isalnum
		Check if character is alphanumeric
	
	isalpha
		Check if character is alphabetic
	isdigit
		Check if character is decimal digit
	
	islower
		Check if character is lowercase letter
	isupper
		Check if character is uppercase letter
	
	Character conversion functions
	
	tolower
		Convert uppercase letter to lowercase
	toupper
		Convert lowercase letter to uppercase
/////////////////////////////////////////////////////////////////////////////////////////////
	std::count
	#include<algorithm>
	Returns the number of elements in the range [first,last) that compare equal to val.
		int mycount = count (myints, myints+8, 10);
	Complexity
	Linear in the distance between first and last: Compares once each element.
/////////////////////////////////////////////////////////////////////////////////////////////
	#include<cmath>
	Rounds x upward, returning the smallest integral value that is not less than x.
	ceil of 2.3 is 3.0
/////////////////////////////////////////////////////////////////////////////////////////////
	#include<cmath>
	Rounds x downward, returning the largest integral value that is not greater than x.
	floor of 2.3 is 2.0
/////////////////////////////////////////////////////////////////////////////////////////////
	std::min_element
	#include<algorithm>
	Returns an iterator pointing to the element with the smallest value in the range [first,last).
	The comparisons are performed using either operator< for the first version, or comp for the second; 
	An element is the smallest if no other element compares less than it. If more than one element 
	fulfills this condition, the iterator returned points to the first of such elements.
		min_element(myints,myints+7)
	Complexity
	Linear in one less than the number of elements compared.
/////////////////////////////////////////////////////////////////////////////////////////////
rl
	#include<algorithm>
	Assigns val to all the elements in the range [first,last).
		fill (myvector.begin(),myvector.begin()+4,5);
	Complexity
	Linear in the distance between first and last: Assigns a value to each element.
/////////////////////////////////////////////////////////////////////////////////////////////
	std::vector::back
	#include<vector>
	Access last element	
	Returns a reference to the last element in the vector.
		v.back()
	Complexity
		Constant
/////////////////////////////////////////////////////////////////////////////////////////////
	std::vector::pop_back
	#include<vector>
	Removes the last element in the vector, effectively reducing the container size by one.
	This destroys the removed element.
		v.pop_back();
	Complexity
	Constant.
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::vector::push_back
	#include<vector>
	Add element at the end
	Adds a new element at the end of the vector, after its current last element. 
	The content of val is copied (or moved) to the new element.
		v.push_back( x );
	Constant (amortized time, reallocation may happen).
	If a reallocation happens, the reallocation is itself up to linear in the entire size.
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::upper_bound
	#include<algorithm>
	Return iterator to upper bound
	Returns an iterator pointing to the first element in the range [first,last) 
	which compares greater than val.		
		upper_bound (v.begin(), v.end(), x )
	if element is not found return iterator to end
	Complexity	
	On average, logarithmic in the distance between first and last: Performs approximately log2(N)+1 element comparisons (where N is this distance).
	On non-random-access iterators, the iterator advances produce themselves an additional linear complexity in N on average.	
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::lower_bound
	#include<algorithm>
	Return iterator to lower bound
	Returns an iterator pointing to the first element in the range [first,last) 
	which does not compare less than val.
		lower_bound (v.begin(), v.end(), x );		
	if element is not found return iterator to end		
	Complexity
	On average, logarithmic in the distance between first and last: Performs approximately log2(N)+1 element comparisons (where N is this distance).
	On non-random-access iterators, the iterator advances produce themselves an additional linear complexity in N on average.
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::binary_search
	#include<algorithm>
	Test if value exists in sorted sequence
	Returns true if any element in the range [first,last) is equivalent to val, and false otherwise.
		binary_search (v.begin(), v.end(), 3)
	Complexity
	On average, logarithmic in the distance between first and last: Performs approximately log2(N)+2 element comparisons (where N is this distance).
	On non-random-access iterators, the iterator advances produce themselves an additional linear complexity in N on average.		
/////////////////////////////////////////////////////////////////////////////////////////////
	std::unique
	#include<algorithm>
	Remove consecutive duplicates in range
		unique ( v.begin(), v.end() ); 
	Complexity
	For non-empty ranges, linear in one less than the distance between first and last: 
	Compares each pair of elements, and possibly performs assignments on some of them.	
/////////////////////////////////////////////////////////////////////////////////////////////		
	std::priority_queue::priority_queue
	#include <queue>
	Constructs a priority_queue container adaptor object.
		int myints[]= {10,60,50,20};	
		priority_queue<int> first;
		priority_queue<int> second (myints,myints+4);
		priority_queue< int, vector<int>, greater<int> > third (myints,myints+4)
		
		priority_queue<int> first;
		first = priority_queue<int>(); // clear the queue
	The example does not produce any output, but it constructs different priority_queue objects:
	First is empty.
	Second contains the four ints defined for myints, with 60 (the highest) at its top.
	Third has the same four ints, but because it uses greater instead of the default (which is less), 
	it has 10 as its top element.		
/////////////////////////////////////////////////////////////////////////////////////////////
	pow
	#include<cmath>
	Returns base raised to the power exponent:
	The result of raising base to the power exponent.
		pow (7.0, 3.0)
	If x is finite negative and y is finite but not an integer value, it causes a domain error.
	if both x and y are zero, it may also cause a domain error.
	If x is zero and y is negative, it may cause a domain error or an overflow range error (or none, depending on the library implementation).	
/////////////////////////////////////////////////////////////////////////////////////////////	
	std::stable_sort
	#include <algorithm>
	Sort elements preserving order of equivalents
	Sorts the elements in the range [first,last) into ascending order, 
	like sort, but stable_sort preserves the relative order of the elements with equivalent values.
	The elements are compared using operator< for the first version, and comp for the second.
/////////////////////////////////////////////////////////////////////////////////////////////
	std::random_shuffle
	Randomly rearrange elements in range
	Rearranges the elements in the range [first,last) randomly.
/////////////////////////////////////////////////////////////////////////////////////////////			
	map insert
	http://www.cplusplus.com/reference/map/map/insert/
/////////////////////////////////////////////////////////////////////////////////////////////			
OJO iteradores begin rbegin etc
/////////////////////////////////////////////////////////////////////////////////////////////			
	sscanf
	char cad[ 1000 ];
	sprintf( cad , "%f" , (double)S );
	REP( i , 10 ) printf( "%c" , cad[ i ] );
	puts( "" );
	printing first digits of a double
/////////////////////////////////////////////////////////////////////////////////////////////			
OJO falta reemplazar insertion sort de algoritmos , esta mal !!!
##########################  strncpy #################################
strncpy(u,t[1024-L+i],L+i+1);//OJo poner el caracter 0 al final
#-----------------------------------------------------------------#

vi inter( vi &a , vi&b ){//intersection of sorted arrays O(n)
	vi c( min( SZ(a) , SZ(b) ) );// SZ(a) + SZ(b) para los demas e-e
	vi :: iterator it = set_intersection( all( a ) , all( b ) , c.begin() );
	c.resize( it - c.begin() );
	return c;
}
//set_union
//The union of two sets is formed by the elements that are present in either one of the sets, or in both. Elements from the second range that have an equivalent element in the first range are not copied to the resulting range.
//The elements in the ranges shall already be ordered according to this same criterion (operator< or comp). The resulting range is also sorted according to this.
//set_difference
//The difference of two sets is formed by the elements that are present in the first set, but not in the second one. The elements copied by the function come always from the first range, in the same order.
//For containers supporting multiple occurrences of a value, the difference includes as many occurrences of a given value as in the first range, minus the amount of matching elements in the second, preserving order.
//The elements in the ranges shall already be ordered according to this same criterion (operator< or comp). The resulting range is also sorted according to this.
//set_intersection
//The intersection of two sets is formed only by the elements that are present in both sets. The elements copied by the function come always from the first range, in the same order.
//The elements in the ranges shall already be ordered according to this same criterion (operator< or comp). The resulting range is also sorted according to this.
//set_symmetric_difference
//The symmetric difference of two sets is formed by the elements that are present in one of the sets, but not in the other. Among the equivalent elements in each range, those discarded are those that appear before in the existent order before the call. The existing order is also preserved for the copied elements.
//The elements in the ranges shall already be ordered according to this same criterion (operator< or comp). The resulting range is also sorted according to this.

// agregar simplex
// arreglar roman numerals
// 3D convex hull
// calculator
ll f( int a , int b ){
	int cntL = 0 , cntR = 0;
	for( int i = b ; i >= a ; --i ){
		if( S[ i ] == '(' ) cntL ++;
		if( S[ i ] == ')' ) cntR ++;
		char c = S[ i ];
		if( cntL == cntR ){
			ll val1 = f( a , i - 1 ) , val2 = f( i + 1 , b );
			if( c == '*' ) return ( val1 * val2 )%MOD;
			if( c == '+' ) return ( val1 + val2 )%MOD;
			if( c == '-' ) return ( val1 - val2 + MOD ) %MOD;
		}
	}
	if( S[ a ] == '(' ) return f( a + 1 , b - 1 );
	if( isdigit( S[ a ] ) ) return S[ a ] - '0';
	return Prime[ S[ a ] - 'a' ];
}
// precision
//http://www.cplusplus.com/reference/cmath/trunc/
//1079 UVA
void doit( ld x ){
	ld fractpart , intpart;
	fractpart = modf( x , &intpart );
	//cout << x << " " << intpart << endl;
	//impr( intpart ) , impr( fractpart );
	int a = (int) round( intpart );
	x = fractpart*60;
	fractpart = modf( x , &intpart );
	int b = (int) round( intpart );
	printf("Case %d: %d:%02d\n", ++tc , a , b );
	/*
	int res = (int)round( 60*x );
	printf("Case %d: %d:", ++tc, res/60);
	if (res % 60 < 10) printf("0%d\n", res%60);
	else printf("%d\n", res%60);
	*/
}
//4661 ACM
int  b;
char s[100], a[20];
sprintf( s, "%.5le", ans );
sscanf( s, "%[^e]e%d", a, &b );
printf( "%s x 10^%d\n", a, b );

/////
Euclid GAme
http://discuss.codechef.com/questions/37284/gameaam-editorial

int G( int a , int b ){
	if( a > b )swap( a , b );
	if( b == 0 ) return 0;
	if( b % a == 0 ) return 1;
	int g = G( b % a , a );
	int q = b/a;
	if( g == 0 ) return q;
	if( g >= q ) return q - 1;
	return q;
}
/*
int g( int a , int b ){
	if( a > b ) swap( a , b );
	if( a == b ) return 0;
	if( b % a == 0 ) return b/a - 1;
	int G = g( b%a , a );
	if( G >= b/a ) return b/a - 1;
	return b/a;
}
*/
// falta knapsack con backtracking
int firstLess( vi v , int x ){
	vector< int > :: reverse_iterator it = upper_bound( rall( v ) , x , greater< int >() );
	if( it != v.rend() ) return *it;
}

//backtracking highly composite numbers COJ 2913
#include<bits/stdc++.h>
using namespace std;

#define sc( x ) scanf( "%d" , &x )
#define REP( i , n ) for( int i = 0 ; i < n ; ++i )
#define clr( t , val ) memset( t , val , sizeof( t ) )

typedef long double ld;
typedef long long ll;
typedef unsigned long long ull;

const int N = 14;
const int EXP = 54;
int P[ N + 1 ];
ull A[ 14 ] = {2 , 3 , 5 , 7 , 11 , 13 , 17 , 19 , 23 , 29 , 31 , 37 , 41 , 43};
ull F[ 14 ][ 55 ];
ull num , ans , L , CA , maxCA;
bool valid( ull a , ull b ){
	ld aa = a , bb = b;
	if( aa * bb > 1e16 ) return 0;
	return a * b <= L;
}
bool valid2( int pos , int exp ){
	ld aa = A[ pos ] , bb = exp;
	if(	pow( aa , bb ) > 1e16 ) return 0;
	return F[ pos ][ exp ] <= L;
}

void f( int pos , int exp ){
	if( pos == N || exp == 0 ) {
		if( CA > maxCA ){
			maxCA = CA;
			ans = num;
		}else if( CA == maxCA ){
			ans = min( ans , num );
		}
		return;
	}
	
	ull temp1 = num;
	ull temp2 = CA;
	f( pos , exp - 1 );
	num = temp1;
	CA = temp2;
	
	temp1 = num;
	temp2 = CA;
	if( valid2( pos , exp ) && valid( num , F[ pos ][ exp ] ) ){
		num *= F[ pos ][ exp ];
		CA *= (exp + 1);
		f( pos + 1 , exp );
	}
	num = temp1;
	CA = temp2;
}
int main(){
	REP( i , N ) F[ i ][ 0 ] = 1;
	REP( i , N ) for( int k = 1 ; k <= EXP ; ++k ) F[ i ][ k ] = F[ i ][ k - 1 ] * A[ i ];
	//REP( i , N ) REP( k , EXP + 1 ) cout << F[ i ][ k ] << char( k == EXP ? 10 : 32 );
	int cases;
	cin >> cases;
	REP( tc , cases ){
		cin >> L;
		maxCA = CA = num = 1;
		ans = 1;
		f( 0 , 54 );
		//
		cout << ans << '\n';
	}
}
/////////////////////////////////////////////////////////////////
//ordinal to arabic
//Project Euler #17: Number to Words
vector< string > S;

string tos( int x ){
	ostringstream out;
	out << x;
	return out.str();
}
string A[] = { "Zero" , "One" , "Two" , "Three" , "Four" , "Five" , "Six" , "Seven" , "Eight" , "Nine" , "Ten" , "Eleven" , "Twelve" , "Thirteen" , "Fourteen" , "Fifteen" , "Sixteen" , "Seventeen" , "Eighteen" , "Nineteen" };
string B[] = { "" , "" , "Twenty" , "Thirty" , "Forty" , "Fifty" , "Sixty" , "Seventy" , "Eighty" , "Ninety" };

string get2( int val ){
	if( val < 20 ) return A[ val ];
	if( val % 10 == 0 ) return B[ val / 10 ];
	return B[ val / 10 ] + " " + A[ val % 10 ];
}
string get( int val ){
	if( val < 100 ) return get2( val );
	int p = val / 100;
	if( val % 100 == 0 ) return A[ p ] + " Hundred";
	return A[ p ] + " Hundred " + get2( val - p * 100 ); 
}
string f( string s ){
	if( s == "0" ) return A[ 0 ];
	int n = SZ( s );
	int p = (n - 1)% 3;
	string ans;
	bool start = 0;
	for( int i = 0 ; i < n ; ++ i ){
		int ind , val = 0;
		for( int j = i ; j < n ; ++j ){
			ind = j;
			val = val * 10 + (s[ j ] - '0');
			if( j % 3 == p ) break;
		}
		if( val ){
			if( start ) ans += " ";
		 	if( S[ n - 1 - i ] != "" ) ans += get( val ) + " " + S[ n - 1 - i ];
		 	else ans += get( val );
		 	start = 1;
		 }
		i = ind;
	}
	return ans;
}
//http://www.merriam-webster.com/table/dict/number.htm
string SS[] = { "" , "Thousand" , "Million" , "Billion" , "Trillion" };
int main(){
	REP( j , 5 ) REP( k , 3 ) S.pb( SS[ j ] );
	int cases;
	cin >> cases ;
	REP( tc , cases ){
		string s;
		cin >> s;
		cout << f( s ) << '\n';
	}
}

/////////////////////////////////////////////////////////////////////////////////////
KAWIGI
general//C:\Dev-Cpp\MinGW32\bin
languages//C:\Dev-Cpp\MinGW32\bin\g++.exe $PROBLEM$.cpp
#-----------------------------------------------------------------#
