#include <bits/stdc++.h>
using namespace std;

#ifdef LOCAL
	const bool local = true;
#else 
	const bool local = false;
#endif

//#pragma GCC optimize("Ofast,unroll-loops")
//#pragma GCC target("avx,avx2,fma")

int iter_CL, iter_CL2, iter_DEADLINE;

inline bool DEADLINE(){
	return clock() / (double) CLOCKS_PER_SEC >= 15.9;
}

inline bool DEADLINE_ITER(){

	iter_DEADLINE++;
	if (iter_DEADLINE & 31) return false;
	return clock() / (double) CLOCKS_PER_SEC >= 8.9;
}

double last_iteration = 10.63;
double last_iteration2 = 5.6;

inline bool CLOCK(){
	return clock() / (double) CLOCKS_PER_SEC >= last_iteration;
}

inline bool CLOCK2(){
	return clock() / (double) CLOCKS_PER_SEC >= last_iteration2;
}

inline bool CL(){
	iter_CL++;
	if (iter_CL & ((1 << 14) - 1)) return false;
	return clock() / (double) CLOCKS_PER_SEC >= last_iteration;
}

inline bool CL2(){
	iter_CL2++;
	if (iter_CL2 & ((1 << 14) - 1)) return false;
	return clock() / (double) CLOCKS_PER_SEC >= last_iteration2;
}

mt19937 rng(47);

inline int random(int l, int r){
	return l + rng() % (r - l + 1);
}

typedef long long LL;
typedef pair<int, int> PII;
typedef vector<int> VI;
#define MP make_pair
#define PB push_back
#define X first
#define Y second

#define FOR(i, a, b) for(int i = (a); i < (b); ++i)
#define RFOR(i, b, a) for(int i = (b) - 1; i >= (a); --i)
#define ALL(a) a.begin(), a.end()
#define SZ(a) (int)((a).size())
#define FILL(a, value) memset(a, value, sizeof(a))
#define debug(a) cerr << #a << " = " << a << endl;

template<typename T> void setmax(T& x, T y) {x = max(x, y);}
template<typename T> void setmin(T& x, T y) {x = min(x, y);}

namespace MCV{
	struct bipartite_matching {
		vector<vector<int>> g;
		vector<int> r, c, vis;
		bipartite_matching(int h, int w, const vector<vector<int>>& G) : g(G), r(h, -1), c(w, -1), vis(h) {}
				
		bool dfs(int i) {
			if (exchange(vis[i], true)) return false;
			for (int j : g[i]) if (c[j] == -1) return r[i] = j, c[j] = i, true;
			for (int j : g[i]) if (dfs(c[j])) return r[i] = j, c[j] = i, true;
			return false;
		}
		int run() {
			while (!DEADLINE_ITER()) {
				fill(begin(vis), end(vis), false);
				bool updated = false;
				for (int i = 0; i < (int)r.size(); ++i) if (r[i] == -1) updated |= dfs(i);
				if (not updated) break;
			}
			return r.size() - count(begin(r), end(r), -1);
		}
	};
		
	pair<vector<int>, vector<int>> mcv_merge(const vector<vector<int>>& graph, int n, int m){
		bipartite_matching matching(n, m, graph);
		int sz = n + m;
		sz -= matching.run();
		vector<int> rev_match(m, -1);
		FOR(i, 0, n) if (matching.r[i] != -1){
			rev_match[matching.r[i]] = i;
		}
		
		vector<char> visited_left(n, false), visited_right(m, false);
		
		function<void(int, int)> dfs = [&](int v, bool left){
			if (!left){
				visited_right[v] = true;
				if (rev_match[v] != -1 && !visited_left[rev_match[v]]){
					dfs(rev_match[v], true);
				}
			}else{
				visited_left[v] = true;
				for(auto to: graph[v]) if (to != matching.r[v] && !visited_right[to]){
					dfs(to, false);
				}
			}
		};
		
		FOR(i, 0, n) if (matching.r[i] == -1 && !visited_left[i]){
			dfs(i, true);
		}		
		
		vector<int> visited_from_left, unvisited_from_right;
		FOR(i, 0, n) if (visited_left[i]){
			visited_from_left.PB(i);
		}
		
		FOR(i, 0, m) if (!visited_right[i]){
			unvisited_from_right.PB(i);
		}
		
		return {visited_from_left, unvisited_from_right};
	}
};

namespace MIS{
	
	namespace FirstFeasibleSolution{
		
		vector<int> greedy_with_small_random(const vector<vector<int>>& g, int F, int G){
			const int V = SZ(g);
			vector<int> ans;
			vector<int> curr(V);
			iota(ALL(curr), 0);
			vector<int> sz(V, 0);
			FOR(i, 0, V) sz[i] = SZ(g[i]);
			vector<char> active(V, 0);
			
			vector<int> pos;
			while(SZ(curr)){
				int Sz = sz[curr[0]];
				for(auto i: curr){
					setmin(Sz, sz[i]);
				}
				
				int mx = (Sz <= F ? Sz : Sz + G);
				pos.clear();
				for(auto i: curr){
					if (sz[i] <= mx){
						pos.PB(i);
					}
				}
				
				int v = pos[rng() % SZ(pos)];
				
				ans.PB(v);
				active[v] = 2;
				for(auto i: g[v]) if (active[i] == 0){
					active[i] = 1;
					for(auto u: g[i]){
						sz[u]--;
					}
				}
				
				FOR(i, 0, SZ(curr)) if (active[curr[i]] != 0){
					swap(curr[i], curr.back());
					curr.pop_back();
					--i;			
				}
			}

			return ans;
		}
				
		vector<int> simple_greedy(const vector<vector<int>>& g){
			const int V = SZ(g);
			vector<int> ans;
			vector<int> curr(V);
			iota(ALL(curr), 0);
			vector<int> sz(V, 0);
			FOR(i, 0, V) sz[i] = SZ(g[i]);
			vector<char> active(V, 0);
			
			while(SZ(curr)){
				int v = curr[0];
				for(auto i: curr){
					if (sz[i] < sz[v]){
						v = i;
					}
				}
				
				ans.PB(v);
				active[v] = 2;
				for(auto i: g[v]) if (active[i] == 0){
					active[i] = 1;
					for(auto u: g[i]){
						sz[u]--;
					}
				}
				
				FOR(i, 0, SZ(curr)) if (active[curr[i]] != 0){
					swap(curr[i], curr.back());
					curr.pop_back();
					--i;			
				}
			}

			return ans;
		}
	};
	
	const int MAX_MAGIC_SIZE = 447;
	void update(vector<vector<int>>& candidates, vector<int>& ans, vector<int> cand){
		candidates.PB(cand);
		if (SZ(ans) < SZ(cand)){
			ans = cand;
		}
		
		if (SZ(candidates) > MAX_MAGIC_SIZE){
			int mn_id = 0;
			FOR(i, 1, SZ(candidates)){
				if (SZ(candidates[i]) < SZ(candidates[mn_id])){
					mn_id = i;
				}
			}
			
			swap(candidates.back(), candidates[mn_id]);
			candidates.pop_back();
		}		
	}
	
	class Solver{
		private:
		vector<vector<int>> g;
		bool final_iteration;
		vector<vector<int>> candidates;		
		int V;		
		
		void merge(vector<int>& L, const vector<int>& R){
			if (DEADLINE()) {
				if (SZ(R) > SZ(L)){
					L = R;
				}
				return;
			}
			
			int n = SZ(L);
			int m = SZ(R);
			vector<int> id(V, -1);		
			FOR(j, 0, m){
				id[R[j]] = j;
			}
			
			vector<vector<int>> graph(n);
			FOR(i, 0, n){
				int v = L[i];
				if (id[v] != -1){
					graph[i].PB(id[v]);
				}
				
				for(auto j: g[v]) if (id[j] != -1){
					graph[i].PB(id[j]);
				}
			}
			
			auto result = MCV::mcv_merge(graph, n, m);
			vector<int> answer;
			for(auto i: result.X){
				answer.PB(L[i]);
			}
			
			for(auto i: result.Y){
				answer.PB(R[i]);
			}
			
			L = answer;
		}	
					
		void optimize(vector<int> solution, const int ITERATIONS_ESCAPING){			
			
			const int MAX_ITERATIONS_FINAL = 7 * V;
			const int MAX_ITERATIONS = 7 * V;
			const int TABOO_SIZE = V;			
			
			vector<int> nei_in_solution(V, 0);
			vector<char> in_solution(V, false);
			vector<int> taboo(V, -1);
			
			for(auto i: solution){
				in_solution[i] = true;
				for(auto j: g[i]){
					nei_in_solution[j]++;
				}
			}
			
			function<bool()> is_local_maximum = [&](){
				
				int iteration_without_find = 0;
				int iteration_without_impovement = 0;
				
				int timer = 0;
				taboo.assign(V, -1);
				bool real_improvement = false;
				const int ITER = (final_iteration ? MAX_ITERATIONS_FINAL : MAX_ITERATIONS);
				int id_in_solution = 0;
				while(true){
					// if (final_iteration && CL()) break;
					// else if (!final_iteration && CL2()) break;
					iteration_without_find++;
					iteration_without_impovement++;
					
					if (iteration_without_impovement > ITER){
						break;
					}
					
					if (iteration_without_impovement > V * 2 && SZ(solution) <= SZ(ans) - 3){
						break;
					}
							
					timer++;
					
					int v = solution[id_in_solution];
					
					for(auto u: g[v]) if (nei_in_solution[u] == 1 && taboo[u] < timer){
						for(auto x: g[v]){
							nei_in_solution[x]--;
						}
						
						for(auto x: g[u]){
							nei_in_solution[x]++;
						}
						
						in_solution[v] = false;
						in_solution[u] = true;
						solution[id_in_solution] = u;
						
						taboo[v] = timer + random(TABOO_SIZE / 2, TABOO_SIZE);
						
						for(auto x: g[v]) if (nei_in_solution[x] == 0 && x != u){
							solution.PB(x);					
							in_solution[x] = true;
							real_improvement = true;
							iteration_without_impovement = 0;
							for(auto to: g[x]){
								nei_in_solution[to]++;
							}
						}						
												
						iteration_without_find = 0;
						break;
					}
					
					++id_in_solution;
					if (id_in_solution >= SZ(solution)){
						id_in_solution %= SZ(solution);
					}					
				}
				
				return real_improvement;
			};
		
			function<void()> escape_local_maximum = [&](){
				
				vector<int> Candidates;
				FOR(i, 0, V) if (!in_solution[i]){
					Candidates.PB(i);
				}				
				
				int id = Candidates[rng() % SZ(Candidates)];
				for(auto i: g[id]){
					if (in_solution[i]){
						in_solution[i] = false;
						for(auto u: g[i]){
							nei_in_solution[u]--;
						}
					}
					
					nei_in_solution[i]++;
				}
				
				in_solution[id] = true;				
				solution.PB(id);
				FOR(i, 0, SZ(solution)) if (!in_solution[solution[i]]){
					swap(solution[i], solution.back());
					solution.pop_back();
					--i;
				}
			};
			
			int it = ITERATIONS_ESCAPING;
			while(true){
				if (!final_iteration){
					it--;
					if (it < 0 || CL2()) break;					
				}
				else if (CL()) break;
				
				while(is_local_maximum()){}
				
				if (SZ(solution) > SZ(ans)){
					it = ITERATIONS_ESCAPING;
				}
				
				update(candidates, ans, solution);
				if (final_iteration && CLOCK()) break;
				if (!final_iteration && CLOCK2()) break;
				int cnt = 5;
				
				while(cnt--){
					escape_local_maximum();
				}
			}
		}
		
		vector<int> _id;	
		public:
		vector<int> ans, mx;
		Solver(){}
				
		Solver(const vector<vector<int>>& _g, bool _final_iteration){
			g = _g;
			final_iteration = _final_iteration;
			
			V = SZ(g);
			if (V > 2){				
					if (!final_iteration){
						FOR(it, 0, 5){		
							auto sol = FirstFeasibleSolution::greedy_with_small_random(g, 1, 0);
							shuffle(ALL(sol), rng);
							update(candidates, ans, sol);
							optimize(sol, 12);						
						}
						
						FOR(it, 0, 5){		
							auto sol = FirstFeasibleSolution::greedy_with_small_random(g, random(2, 7), random(0, 3));
							shuffle(ALL(sol), rng);
							update(candidates, ans, sol);
							optimize(sol, 12);						
						}
						
						auto sol = FirstFeasibleSolution::simple_greedy(g);
						optimize(sol, 12);	
					}
					else{					
						last_iteration = 11.8;	
						auto sol = FirstFeasibleSolution::simple_greedy(g);
						optimize(sol, 12);	
						sol = FirstFeasibleSolution::greedy_with_small_random(g, 1, 0);
						last_iteration = 15.5;
						optimize(sol, 12);													
					}
			}else{
				ans = {0};
			}
			
			_id.resize(SZ(candidates));
			iota(ALL(_id), 0);
			mx = ans;
		}				
				
		void solve(){
			if (V <= 2){
				return;
			}else{
				if (DEADLINE()) return;
				for(auto i: _id){
					 merge(mx, candidates[i]);
				}
					 
				shuffle(ALL(_id), rng);
				if (SZ(ans) < SZ(mx)){
					ans = mx;
				}
				
				mx = ans;				
			}			
		}
	};	
};

namespace Preparation{
	vector<vector<int>> solve(vector<vector<vector<int>>>& allGraphs){
		const int sz = SZ(allGraphs);
		if (sz == 0){
			return {};
		}
		
		vector<int> idx(sz);
		iota(ALL(idx), 0);
		sort(ALL(idx), [&](int u, int v){return SZ(allGraphs[u]) < SZ(allGraphs[v]);});
		vector<vector<int>> ans(sz);
		vector<MIS::Solver> solvers(sz);
		vector<int> sizes(sz);
		
		FOR(i, 0, sz){
			sizes[i] = SZ(allGraphs[idx[i]]);
		}			
		
		FOR(i, 0, sz){
			solvers[i] = MIS::Solver(allGraphs[idx[i]], i == sz - 1);					
		}		
		
		int iter = 0;
		while(!DEADLINE()){
			iter++;
			if (iter == 3){
				break;
			}
			
			RFOR(i, sz, 0){
				solvers[i].solve();
			}
		}
		
		if (sz > 1 && sizes[sz - 2] * 1.5 >= sizes[sz - 1]){
			while(!DEADLINE()){
				solvers.back().solve();
				if (DEADLINE()) break;
				solvers[sz - 2].solve();
			}
		}else{
			while(!DEADLINE()){
				solvers.back().solve();
			}		
		}
		
		FOR(i, 0, sz){
			ans[idx[i]] = solvers[i].ans;
		}
		
		return ans;
	};
};

namespace Constants{
	vector<vector<int>> g;
	vector<int> ans;
	const int H = 300;
	const int offset = 10;
	int s[H][H];
	int h, w, n, m, t;
	int need_sum_per_rect;
	int max_ans;
	int rectangle_id[H][H][2];
	int V;
	const int MAX_V = 200007;
};
	
namespace BipartiteMatching{

	vector<int> matching(vector<vector<int>>& G, int n, int m){
		MCV::bipartite_matching match(n, m, G);
		match.run();
		vector<int> mt(m, -1);
		FOR(i, 0, n) if (match.r[i] != -1){
			mt[match.r[i]] = i;
		}
		
		return mt;
	}
};

namespace Shredder{
	using namespace Constants;
	
	bool used[MAX_V];
	int mp[MAX_V];
	int vertices[MAX_V], ptr_vertices;
	vector<vector<vector<int>>> allGraphs;
	vector<vector<int>> allVertices;
	
	void solve(){
 
		FOR(i, 0, V) if (!used[i]){
			ptr_vertices = 0;
			queue<int> q;
			q.push(i);
			
			vertices[ptr_vertices] = i;
			mp[i] = ptr_vertices;
			ptr_vertices++;			
			used[i] = true;
			
			while(!q.empty()){
				int v = q.front();
				q.pop();
				
				for(auto u: g[v]) if (!used[u]){
					vertices[ptr_vertices] = u;
					mp[u] = ptr_vertices;
					ptr_vertices++;
					used[u] = true;		
					q.push(u);			
				}
			}
			
			vector<vector<int>> G(ptr_vertices);
			FOR(I, 0, ptr_vertices){
				int v = vertices[I];
				G[I].reserve(SZ(g[v]));
				for(auto j: g[v]){
					G[I].PB(mp[j]);
				}
			}
					
			allGraphs.PB(G);
			allVertices.PB(vector<int>(vertices, vertices + ptr_vertices));
		}		
		
		auto res = Preparation::solve(allGraphs);
		FOR(i, 0, SZ(res)){
			for(auto v: res[i]){
				ans.PB(allVertices[i][v]);
			}
		}
	}
};


namespace Input{
	using namespace Constants;
    
    void read_graph(char *argv[]) {
        freopen(argv[1], "r", stdin);
        scanf("%d %d", &n, &m);
        g.assign(n, VI());
        for (int i = 0; i < m; ++i) {
            int u, v;
            scanf("%d %d", &u, &v);
            --u, --v;
            g[u].PB(v);
            g[v].PB(u);
        }        
        V = n;
    }
};

namespace Output{
	using namespace Constants;
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

	void write(){
		printf("ok: %d\n", check_ans(g, ans));
		printf("%d\n", SZ(ans));
	}
};

int main(int argc, char* argv[]){
	Input::read_graph(argv);
	Shredder::solve();
	Output::write();
	return 0;
}

