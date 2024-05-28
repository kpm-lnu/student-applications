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

class pcg32 
{
    uint64_t       state      = 0x4d595df4d0f33173;
    static uint64_t const multiplier = 6364136223846793005u;
    static uint64_t const increment  = 1442695040888963407u;
public:
 
    using result_type = uint32_t;
 
    explicit pcg32(result_type seed)
    {
        state = seed + increment;
        operator()();
    }
 
    static constexpr result_type max() { return UINT_MAX; }
 
    static constexpr result_type min() { return 0; }

    static uint32_t rotr32(uint32_t x, unsigned r)
    {
        return x >> r | x << (-r & 31);
    }
 
    result_type operator() () {
        uint64_t x = state;
        unsigned count = (unsigned)(x >> 59);

        state = x * multiplier + increment;
        x ^= x >> 18;
        return rotr32((uint32_t)(x >> 27), count);
    }
};

pcg32 rnd(time(0));

VI solve(const vector<VI>& g)
{
    if (g.empty())
        return {};
    Perf _("solve");
    double temperature = 40;
    const double alpha = 0.9925;
    int current_e = 0;
    VI taken(SZ(g));
    VI taken_neigs(SZ(g));
    const int iters_count = 2000;
    const int same_temp_count = 1.5e3;
    FOR(i, 0, iters_count)
    {
        FOR(j, 0, same_temp_count)
        {
            int u = rnd() % SZ(g);
            int delta_e = 1 - taken_neigs[u];
            if (!taken[u])
                delta_e = -delta_e;
            // dout("delta_e", delta_e);
            double p = exp(-delta_e / temperature);
            if (delta_e < 0 || rnd() < UINT_MAX * p)
            {
                current_e += delta_e;
                for (auto v : g[u])
                    if (taken[u])
                        taken_neigs[v]--;
                    else
                        taken_neigs[v]++;
                taken[u] ^= 1;
            }
        }
        int sum = 0;
        for (int u = 0; u < SZ(g); ++u)
        {
            if (taken[u])
                sum += taken_neigs[u] - 2;
        }
        assert(current_e == sum / 2);
        // cout << "current_e = " <<  current_e <<  " sum = "  << sum / 2 << "\n";
        temperature *= alpha;
    }
    int j = 0;
    VI perm(SZ(taken));
    FOR(i, 0, SZ(perm)) perm[i] = i;
    shuffle(ALL(perm), rnd);
    while (true)
    {    
        for (auto u : perm)
        {
            int delta_e = 1 - taken_neigs[u];
            if (!taken[u])
                delta_e = -delta_e;
            if (delta_e <= 0)
            {
                current_e += delta_e;
                for (auto v : g[u])
                    if (taken[u])
                        taken_neigs[v]--;
                    else
                        taken_neigs[v]++;
                taken[u] ^= 1;
            }
        }
        int sum = 0;
        for (int u = 0; u < SZ(g); ++u)
        {
            if (taken[u])
                sum += taken_neigs[u] - 2;
        }
        assert(current_e == sum / 2);
        // cout << "current_e = " <<  current_e <<  " sum = "  << sum / 2 << "\n";
        j++;
        if (j % 300)
            if ((double)clock() / CLOCKS_PER_SEC >= 3.8)
                break;
    }
    dout("end temperature", temperature);
    VI ans;
    FOR(u, 0, SZ(g))
    {
        if (taken[u])
        {
            bool ok = true;
            for (auto v : g[u])
            {
                if (taken[v])
                    ok = false;
            }
            if (!ok)
            {
                // dout("warning: not ok");
                taken[u] = false;
            }
            else
            {
                ans.PB(u);
            }
        }
    }
    dout("current_e", current_e);
    return ans;
}

bool check_ans(const vector<VI>& g, const VI& ans)
{
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

