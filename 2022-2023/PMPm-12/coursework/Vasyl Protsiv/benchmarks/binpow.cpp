#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <random>
#include <cstring>
using namespace std;

const int mod = 1e9 + 7;

void dout() { cerr << endl; }

template <typename Head, typename... Tail>
void dout(Head H, Tail... T) {
    cerr << H << ' ';
    dout(T...);
}

struct Perf
{
    typedef std::chrono::high_resolution_clock Clock;
    Perf(string n) : name(std::move(n)), start(Clock::now()) {}
    ~Perf() {
        // dout(name, chrono::duration_cast<chrono::microseconds>(Clock::now() - start).count());
        dout(name, chrono::duration_cast<chrono::milliseconds>(Clock::now() - start).count(), "ms");
    }
    string name;
    std::chrono::time_point<Clock> start;
};


vector<vector<int>> mul(vector<vector<int>> a, vector<vector<int>> b)
{
    int n = a.size();
    vector<vector<int>> c(n, vector<int>(n));
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
            {
                c[i][j] = (c[i][j] + (long long)a[i][k] * b[k][j]) % mod;
            }
        }
    }
    return c;
}

vector<vector<int>> binpow(vector<vector<int>> a, int k)
{
    int n = a.size();
    vector<vector<int>> res(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        res[i][i] = 1;
    while (k > 0)
    {
        if (k % 2 != 0)
            res = mul(res, a);
        a = mul(a, a);
        k /= 2;
    }
    return res;
}

int solve(vector<vector<int>> g, int k)
{
    Perf _("binpow");
    int n = g.size();
    auto dp = binpow(g, k);
    return dp[0][n - 1];
}

int solveNaive(vector<vector<int>> gg, int k)
{
    Perf _("naive");
    int n = 100;
    // array<array<int, 100>, 100> g;
    // array<array<int, 100>, 100> dp[2];
    int g[100][100];
    int dp[2][100];
    memset(dp, 0, sizeof dp);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            g[i][j] = gg[i][j];
    }
    dp[0][0] = 1;
    // array<array<int, 100>, 100> ndp;
    // vector<vector<int>> dp = g;
    // vector<vector<int>> ndp;
    for (int i = 0; i < k; i++)
    {
        // ndp.assign(n, vector<int>(n));
        memset(dp[(i + 1) % 2], 0, sizeof dp[0]);
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                dp[(i + 1) % 2][u] = (dp[(i + 1) % 2][u] + (long long)dp[i % 2][v] * g[v][u]) % mod;
            }
        }
        // swap(ndp, dp);
    }
    return dp[k % 2][n - 1];
}


void test()
{
    mt19937 rnd(47);
    auto rn = [&](int l, int r)
    {
        return uniform_int_distribution<int>(l, r)(rnd);
    };
    for (int _ = 0; _ < 1000; _++)
    {
        int n = 100;
        int m = 10000;
        int k = 10000;
        // int n = rn(1, 100);
        // int m = rn(1, 1000);
        // int k = rn(1, 1000);
        vector<vector<int>> g(n, vector<int>(n));
        for (int i = 0; i < m; i++)
        {
            int u = rn(0, n - 1);
            int v = rn(0, n - 1);
            g[u][v]++;
        }
        auto ans = solve(g, k);
        auto ansNaive = solveNaive(g, k);
        assert(ans == ansNaive);     
    }
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    test();
    int n, m, k;
    cin >> n >> m >> k;
    vector<vector<int>> g(n, vector<int>(n));
    for (int i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;
        g[u - 1][v - 1]++;
    }
    cout << solveNaive(g, k) << "\n";
    return 0;
}
