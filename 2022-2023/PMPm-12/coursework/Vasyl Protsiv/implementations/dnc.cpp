#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

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

int solve(vector<int> a, int m)
{
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

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    cout << solve(a, k) << "\n";
    return 0;
}
