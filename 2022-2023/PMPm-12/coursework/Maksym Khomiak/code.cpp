#include <bits/stdc++.h>
using namespace std;

//#pragma GCC optimize("Ofast")
//#pragma GCC target("sse4")

#define FOR(i, a, b) for(int i = (a); i < (b); ++i)
#define RFOR(i, b, a) for(int i = (b) - 1; i >= (a); --i)

#define SZ(a) (int)((a).size())
#define ALL(a) a.begin(), a.end()

typedef long long LL;
typedef pair<int, int> PII;

const int MAXN = 807;
const int MAXM = 20'007;
const int MAXK = 57;
const int MAXCDEST = 7;
const int MAXPATHCOUNT = 5;
const int MAXLEN = 207;

int a[MAXK];
double b[MAXK][2];
int u[MAXM], v[MAXM], cnt[MAXM];
double cap[MAXM], util[MAXM];
int dest[MAXCDEST], path_count[MAXCDEST];
double amount[MAXCDEST];
int len[MAXCDEST][MAXPATHCOUNT];
int path[MAXCDEST][MAXPATHCOUNT][MAXLEN];

int usedEdge[MAXM]; // скільки разів ребро зустрічається в шляхах поточного агента

const int P = 100;

void setmax(double& x, double val)
{
  x = max(x, val);
}

double getD(int q, int p)
{
  double res = 0;
  FOR(l, 0, len[q][p])
  {
    int e = path[q][p][l];
    setmax(res, util[e] / cap[e]);
  }
  return res;
}

int main()
{
  //ios_base::sync_with_stdio(0), cin.tie(0), cout.tie(0);
  int n, m, k, t, agentIdx, cDest;
  scanf("%d%d%d", &n, &m, &k);
  FOR(i, 0, k)
  {
    scanf("%d%lf%lf", a + i, b[i], b[i] + 1);
    a[i]--;
  }
  FOR(i, 0, m)
  {
    scanf("%d%d%lf%lf%d", u + i, v + i, cap + i, util + i, cnt + i);
    u[i]--;
    v[i]--;
  }
  scanf("%d%d%d", &t, &agentIdx, &cDest);
  agentIdx--;
  LL bAg[2];
  FOR(i, 0, 2)
    bAg[i] = b[agentIdx][i];
  FOR(q, 0, cDest)
  {
    scanf("%d%d%lf", dest + q, path_count + q, amount + q);
    dest[q]--;
    FOR(p, 0, path_count[q])
    {
      scanf("%d", len[q] + p);
      FOR(l, 0, len[q][p])
      {
        scanf("%d", path[q][p] + l);
        path[q][p][l]--;
        usedEdge[path[q][p][l]]++;
      }
    }
  }
  ....
  double x0[MAXCDEST][MAXPATHCOUNT];
  FOR(q, 0, cDest)
  {
    if(t == 1)
    {
      FOR(p, 0, path_count[q])
        x0[q][p] = 1.0 / path_count[q];
    }
    else
    {
      double sumX = 0;
      FOR(p, 0, path_count[q])
      {
        x0[q][p] = 0;
        bool wasZeroCnt = false;
        FOR(l, 0, len[q][p])
        {
          int e = path[q][p][l];
          wasZeroCnt |= cnt[e] == 0;
          if(!wasZeroCnt)
          {
            x0[q][p] += util[e] / cnt[e] / usedEdge[e];
          }
        }
        x0[q][p] /= len[q][p];
        if(wasZeroCnt)
        {
          x0[q][p] = 0;
        }
        sumX += x0[q][p];
      }
      FOR(p, 0, path_count[q])
      {
        x0[q][p] /= sumX;
      }
      FOR(j, 0, 3)
      {
        double coef = (j == 2 ? P - (bAg[0] % P) - (bAg[1] % P) : bAg[j] % P) / (double)P;
        int le = j * path_count[q] / 3, ri = (j + 1) * path_count[q] / 3;
        if(le < ri)
        {
          double s = 0;
          FOR(p, le, ri)
          {
            s += x0[q][p];
          }
          if(s < 1e-9)
          {
            FOR(p, le, ri)
            {
              x0[q][p] = 0.01;
              s += x0[q][p];
            }
          }
          FOR(p, le, ri)
            x0[q][p] *= coef / s;
        }
      }
    }
    FOR(j, 0, 2)
      bAg[j] /= P;
  }
  double x[MAXCDEST][MAXPATHCOUNT], nx[MAXCDEST][MAXPATHCOUNT];
  FOR(q, 0, cDest)
    FOR(p, 0, path_count[q])
      x[q][p] = nx[q][p] = x0[q][p];
  double k1 = 0.003;
  for(int it = 0; it < 100000 && k1 > 1e-5; it++)
  {
    double mlu = 0;
    FOR(q, 0, cDest)
    {
      double minD, maxD = 0;
      int minPath = 0, maxPath = 0;
      FOR(p, 0, path_count[q])
      {
        double D = getD(q, p);
        if(p == 0 || D < minD)
        {
          minD = D;
          minPath = p;
        }
        if(p == 0 || D > maxD)
        {
          maxD = D;
          maxPath = p;
        }
      }
      setmax(mlu, maxD);
      double delta = k1 * nx[q][maxPath];
      nx[q][maxPath] -= delta;
      nx[q][minPath] += delta;
    }
    FOR(q, 0, cDest)
      FOR(p, 0, path_count[q])
      {
        double dutil = (nx[q][p] - x[q][p]) * amount[q];
        FOR(l, 0, len[q][p])
          util[path[q][p][l]] += dutil;
      }
    double nmlu = 0;

    FOR(q, 0, cDest)
      FOR(p, 0, path_count[q])
        setmax(nmlu, getD(q, p));
    if(nmlu < (1 - 1e-6) * mlu)
    {
      FOR(q, 0, cDest)
        FOR(p, 0, path_count[q])
          x[q][p] = nx[q][p];
    }
    else
    {
      FOR(q, 0, cDest)
        FOR(p, 0, path_count[q])
        {
          double dutil = (nx[q][p] - x[q][p]) * amount[q];
          FOR(l, 0, len[q][p])
            util[path[q][p][l]] -= dutil;
        }
      k1 *= 0.5;
    }
  }
  double k2 = t <= 10 ? 0.7 : (t <= 20 ? 0.25 : 0.1);
  FOR(q, 0, cDest)
  {
    FOR(p, 0, path_count[q])
    {
      x[q][p] = (1 - k2) * x0[q][p] + k2 * x[q][p];
      printf("%.15lf ", x[q][p] * amount[q]);
    }
    printf("\n");
  }
  LL pwP = 1;
  LL bOut[2] = {0, 0};
  FOR(q, 0, cDest)
  {
    FOR(j, 0, 2)
    {
      double s = 0;
      FOR(p, j * path_count[q] / 3, (j + 1) * path_count[q] / 3)
        s += x[q][p];
      bOut[j] += min(P - 1, (int)(s * P)) * pwP;
    }
    pwP *= P;
  }
  printf("%lld %lld\n", bOut[0], bOut[1]);
  return 0;
}