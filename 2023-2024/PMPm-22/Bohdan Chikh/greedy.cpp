#pragma GCC optimize("Ofast")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
#define NDEBUG
#include <bits/stdc++.h>
using namespace std;
using LL = long long;
using ULL = unsigned long long;
using VI = vector<int>;
using VL = vector<LL>;
using PII = pair<int, int>;
using PLL = pair<LL, LL>;

#define SZ(a) (int)a.size()
#define ALL(a) a.begin(), a.end()
#define MP make_pair
#define PB push_back
#define EB emplace_back
#define F first
#define S second
#define FOR(i, a, b) for (int i = (a); i<(b); ++i)
#define RFOR(i, b, a) for (int i = (b)-1; i>=(a); --i)
#define FILL(a, b) memset(a, b, sizeof(a))

void dout() { cerr << endl; }

template <typename Head, typename... Tail>
void dout(Head H, Tail... T) {
    cerr << H << ' ';
    dout(T...);
}

#ifdef NDEBUG
#define dout(...) 47
#endif

const int INF = 1e9 + 47;

struct Perf
{
    using clock = chrono::high_resolution_clock;
    using time_point = chrono::time_point<clock>;
    Perf(const string& s) : start(clock::now()), name(s) {}
    ~Perf()
    {
        time_point end = clock::now();
        auto dur = chrono::duration_cast<chrono::milliseconds>(end - start);
        dout(name, dur.count(), "ms");
    }
    time_point start;
    string name;
};

VI solve(const vector<VI>& g)
{
    mt19937 rnd(74);
    if (g.empty())
        return {};
    Perf _("solve");

    int n = SZ(g);
    VI deg(n);
    vector<PII> vec;
    for (int i = 0; i < SZ(g); i++)
    {
        deg[i] = SZ(g[i]);
        vec.EB(deg[i], i);
    }

    
    sort(ALL(vec), greater<PII>());
    VI ans, color(n, 0);
    while (true)
    {
        VI candidates;
        for (int u = 0; u < n; u++)
        {
            if (color[u] == 0)
            {
                if (!candidates.empty() && deg[u] < deg[candidates[0]])
                    candidates.clear();
                if (candidates.empty() || deg[u] == deg[candidates[0]])
                    candidates.PB(u);
            }
        }

        if (candidates.empty())
            break;
        int u = candidates[uniform_int_distribution<int>(0, SZ(candidates) - 1)(rnd)];

        color[u] = 1;
        ans.PB(u);
        for (auto v : g[u])
        {
            if (color[v] == 0)
            {
                for (auto w : g[v])
                {
                    if (color[w] == 0)
                    {
                        --deg[w];
                        assert(deg[w] >= 0);
                    }
                }
            }
            color[v] = 1;
        } 
    }    

    return ans;
}

bool check_ans(const vector<VI>& g, const VI& ans)
{
    int n = SZ(g);
    bool ok = true;
    VI used(n, 0);
    for (auto u : ans)
    {
        used[u] = 1;
        for (auto v : g[u])
            if (used[v])
                ok = false;
    }
    return ok;
}

void solve(char* argv[])
{
    // Perf _("All");
    freopen(argv[1], "r", stdin);
    int n, m;
    cin >> n >> m;
    vector<VI> g(n);
    for (int i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;
        --u, --v;
        g[u].PB(v);
        g[v].PB(u);
    }
    
    auto ans = solve(g);
    cout << "ok: " << check_ans(g, ans) << "\n";
    cout << SZ(ans) << "\n";
}

int main(int argc, char *argv[]) 
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    solve(argv);
    return 0;
}

