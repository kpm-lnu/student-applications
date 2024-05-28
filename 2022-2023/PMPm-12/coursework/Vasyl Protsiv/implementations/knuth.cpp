#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
using LL = long long;
 
const LL INF = 1e16 + 47;

LL solve(vector<int> x)
{
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
 
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    int n;
    cin >> n;
    vector<int> x(n);
    for (int i = 0; i < n; i++)
    {
        cin >> x[i];
    }
    cout << solve(x) << "\n";
    return 0;
}