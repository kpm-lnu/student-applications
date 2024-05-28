#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int mod = 1e9 + 7;

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
    int n = g.size();
    auto dp = binpow(g, k);
    return dp[0][n - 1];
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, k;
    cin >> n >> m >> k;
    vector<vector<int>> g(n, vector<int>(n));
    for (int i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;
        g[u - 1][v - 1]++;
    }
    cout << solve(g, k) << "\n";
    return 0;
}
