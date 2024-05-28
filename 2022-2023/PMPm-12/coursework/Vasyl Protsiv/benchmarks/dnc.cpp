#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <random>
#include <cassert>
#include <chrono>
using namespace std;

void dout() { cerr << endl; }

template <typename Head, typename... Tail>
void dout(Head H, Tail... T) {
    cerr << H << ' ';
    dout(T...);
}

vector<int> dp;
vector<int> prev_dp;
vector<vector<int>> cost;

void compute_dp(int l, int r, int opt_l, int opt_r)
{
    if (r < l)
        return;
    int x = (l + r) / 2;
    int opt_x = opt_l;
    for (int j = opt_l; j <= min(x, opt_r); j++)
    {
        int val = prev_dp[j - 1] + cost[j - 1][x];
        if (val > dp[x])
        {
            dp[x] = val;
            opt_x = j;
        }
    }
    compute_dp(l, x - 1, opt_l, opt_x);
    compute_dp(x + 1, r, opt_x, opt_r);
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

int solveNaive(vector<int> a, int m)
{
    Perf _("naive");
    int n = a.size();
    auto b = a;
    sort(b.begin(), b.end());
    b.erase(unique(b.begin(), b.end()), b.end());
    for (auto& x : a)
    {
        x = lower_bound(b.begin(), b.end(), x) - b.begin();
    }
    cost.assign(n + 1, vector<int>(n + 1));
    for (int l = 0; l < n; l++)
    {
        vector<int> cnt(n);
        int cur = 0;
        for (int r = l; r < n; r++)  
        {
            if (cnt[a[r]] == 0)
                cur++;
            cnt[a[r]]++;
            cost[l][r + 1] = cur;
        } 
    }
    dp.assign(n + 1, 0);
    for (int i = 1; i <= m; i++)
    {
        swap(dp, prev_dp);
        dp.assign(n + 1, 0);
        for (int j = 1; j <= n; j++)
        {
            for (int k = i; k <= j; k++)
            {
                dp[j] = max(dp[j], prev_dp[k - 1] + cost[k - 1][j]);
            }
        }
    }
    return dp[n];
}

int solve(vector<int> a, int m)
{
    Perf _("optimized");
    int n = a.size();
    auto b = a;
    sort(b.begin(), b.end());
    b.erase(unique(b.begin(), b.end()), b.end());
    for (auto& x : a)
    {
        x = lower_bound(b.begin(), b.end(), x) - b.begin();
    }
    cost.assign(n + 1, vector<int>(n + 1));
    for (int l = 0; l < n; l++)
    {
        vector<int> cnt(n);
        int cur = 0;
        for (int r = l; r < n; r++)  
        {
            if (cnt[a[r]] == 0)
                cur++;
            cnt[a[r]]++;
            cost[l][r + 1] = cur;
        } 
    }
    dp.assign(n + 1, 0);
    for (int i = 1; i <= m; i++)
    {
        swap(dp, prev_dp);
        dp.assign(n + 1, 0);
        compute_dp(1, n, 1, n);
    }
    return dp[n];
}

mt19937 rnd(47);

void test(int n, int m)
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
        auto ans = solve(a, m);
        auto ansNaive = solveNaive(a, m);
        assert(ans == ansNaive);
    }
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    test(2000, 20);
    // int n, k;
    // cin >> n >> k;
    // vector<int> a(n);
    // for (int i = 0; i < n; i++)
    // {
    //     cin >> a[i];
    // }
    // cout << solve(a, k) << "\n";
    return 0;
}
