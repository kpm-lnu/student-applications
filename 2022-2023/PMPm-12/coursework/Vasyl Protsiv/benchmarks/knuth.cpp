#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <random>
#include <cassert>
#include <chrono>
using namespace std;
using LL = long long;

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
        dout(name, chrono::duration_cast<chrono::microseconds>(Clock::now() - start).count());
    }
    string name;
    std::chrono::time_point<Clock> start;
};
 
const LL INF = 1e16 + 47;

LL solveNaive(vector<int> x)
{
    Perf _("naive");
    int n = x.size();
    vector<vector<LL>> dp(n, vector<LL>(n));
    for (int l = n - 1; l >= 0; l--)
    {
        LL sum = x[l];
        for (int r = l + 1; r < n; r++)
        {
            sum += x[r];
            dp[l][r] = sum;
            LL mn = INF;
            for (int k = l; k < r; k++)
            {
                if (dp[l][k] + dp[k + 1][r] < mn)
                {
                    mn = dp[l][k] + dp[k + 1][r];
                }
            }
            dp[l][r] += mn;
        }
    }
    return dp[0][n - 1];
}

LL solve(vector<int> x)
{
    Perf _("optimized");
    int n = x.size();
    vector<vector<LL>> dp(n, vector<LL>(n));
    vector<vector<int>> opt(n, vector<int>(n, -1));
    for (int i = 0; i < n; i++)
    {
        opt[i][i] = i;
    }
    for (int l = n - 1; l >= 0; l--)
    {
        LL sum = x[l];
        for (int r = l + 1; r < n; r++)
        {
            sum += x[r];
            dp[l][r] = sum;
            LL mn = INF;
            int mn_k = opt[l][r - 1];
            for (int k = opt[l][r - 1]; k <= opt[l + 1][r]; k++)
            {
                if (k < r && dp[l][k] + dp[k + 1][r] < mn)
                {
                    mn = dp[l][k] + dp[k + 1][r];
                    mn_k = k;
                }
            }
            dp[l][r] += mn;
            opt[l][r] = mn_k;
        }
    }
    return dp[0][n - 1];
}

mt19937 rnd(47);

void test(int n)
{
    auto rn = [&](int l, int r)
    {
        return uniform_int_distribution<int>(l, r)(rnd);
    };
    for (int _ = 0; _ < 1000; _++)
    {
        vector<int> a(n);
        for (int i = 0; i < n; i++)
        {
            a[i] = rn(1, 1'000'000'000);
        }
        auto ans = solve(a);
        auto ansNaive = solveNaive(a);
        assert(ans == ansNaive);
    }
}
 
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    test(1000);
    // int n;
    // cin >> n;
    // vector<int> x(n);
    // for (int i = 0; i < n; i++)
    // {
    //     cin >> x[i];
    // }
    // cout << solve(x) << "\n";
    return 0;
}