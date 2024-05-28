#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
using LL = long long;

struct Line
{
    LL k;
    LL b;
 
    Line(LL x, LL y) : k(x), b(y) {}
 
    LL get(LL x)
    {
        return k * x + b;
    }
};

LL solve(vector<tuple<LL, LL, LL>> v)
{
    int n = v.size() - 1;
    sort(v.begin(), v.end());
    vector<LL> dp(n + 1);
    vector<Line> hull = {{0, 0}};
    const LL inf = 1e18;
    vector<LL> from = {-inf};
    auto intersect = [](Line l, Line r)
    {
        if (r.b - l.b < 0)
            return -(-(r.b - l.b) / (l.k - r.k));
        else
            return (r.b - l.b) / (l.k - r.k);
    };
    int pos = 0;
    for (int i = 1; i <= n; i++)
    {
        auto [x, y, a] = v[i];
        while (pos < from.size() && from[pos] <= -y)
            pos++;
        dp[i] = x * y - a + hull[pos - 1].get(-y);
        Line line(x, dp[i]);
        while (!hull.empty() && intersect(hull.back(), line) <= from.back())
        {
            hull.pop_back();
            from.pop_back();
        }
        if (hull.empty())
            from.push_back(-inf);
        else
            from.push_back(intersect(hull.back(), line));
        hull.push_back(line);
    }
    return *max_element(dp.begin(), dp.end());
}
 
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<tuple<LL, LL, LL>> v(n + 1);
    for (int i = 1; i <= n; i++)
    {
        LL x, y, a;
        cin >> x >> y >> a;
        v[i] = make_tuple(x, y, a);
    }
    cout << solve(v) << "\n";
    return 0;
}