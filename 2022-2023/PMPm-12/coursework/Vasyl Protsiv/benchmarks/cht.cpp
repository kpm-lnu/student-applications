#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <random>
#include <cassert>
#include <chrono>
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

LL solveNaive(vector<tuple<LL, LL, LL>> v)
{
	Perf _("naive");
	int n = v.size() - 1;
	sort(v.begin(), v.end());
	vector<LL> dp(n + 1);
	for (int i = 1; i <= n; i++)
	{
		auto [xi, yi, ai] = v[i];
		for (int j = 0; j < i; j++)
		{
			auto [xj, yj, aj] = v[j];
			dp[i] = max(dp[i], dp[j] + yi * (xi - xj) - ai);
		}
	}
	return *max_element(dp.begin(), dp.end());
}

LL solve(vector<tuple<LL, LL, LL>> v)
{
	Perf _("cht");
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

void test(int n)
{
	mt19937 rnd(47);
	auto rn = [&](LL l, LL r)
	{
		return uniform_int_distribution<LL>(l, r)(rnd);
	};
	for (int _ = 0; _ < 1000; _++)
	{
		set<LL> xs, ys;
		for (int i = 1; i <= n; i++)
		{
			LL x, y;
			do {
				x = rn(1, 1'000'000'000);
			} while (xs.count(x));
			xs.insert(x);
			do {
				y = rn(1, 1'000'000'000);
			} while (ys.count(y));
			ys.insert(y);
		}
		vector<tuple<LL, LL, LL>> v(n + 1);
		auto xit = xs.begin();
		auto yit = xs.rbegin();
		for (int i = 1; i <= n; i++)
		{
			LL x = *xit;
			LL y = *yit;
			LL a = rn(1, x * y);
			xit++;
			yit++;
			v[i] = make_tuple(x, y, a);
		}
		shuffle(v.begin(), v.end(), rnd);
		auto ans = solve(v);
		auto ansNaive = solveNaive(v);
		assert(ans == ansNaive);		
	}
}
 
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    test(100'000);
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