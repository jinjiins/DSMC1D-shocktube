#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct Rng {
    std::mt19937_64 g;
    std::uniform_real_distribution<double> u{0.0, 1.0};
    std::normal_distribution<double> n{0.0, 1.0};
    explicit Rng(uint64_t s) : g(s) {}
    double uni() { return u(g); }
    double nor() { return n(g); }
};

struct Particles {
    std::vector<double> x, vx, vy, vz;
    void add(double x_, double a, double b, double c) { x.push_back(x_); vx.push_back(a); vy.push_back(b); vz.push_back(c); }
    size_t size() const { return x.size(); }
    void remove(size_t i) {  // swap-remove
        size_t l = x.size() - 1;
        x[i] = x[l]; vx[i] = vx[l]; vy[i] = vy[l]; vz[i] = vz[l];
        x.pop_back(); vx.pop_back(); vy.pop_back(); vz.pop_back();
    }
};

using Attrs = std::map<std::string, std::string>;

static Attrs parse_attrs(const std::string& s) {
    static const std::regex re(R"((\w+)\s*=\s*"([^"]*)\")");
    Attrs m;
    for (std::sregex_iterator it(s.begin(), s.end(), re), end; it != end; ++it) m[(*it)[1]] = (*it)[2];
    return m;
}

// Minimal reader for gas.xml: <species id=... ...> <vhs .../> <vss .../> </species>.
// Returns the species attributes merged with those of its <model> child.
static bool load_gas(const std::string& file, const std::string& id, const std::string& model, Attrs& out) {
    std::ifstream f(file);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", file.c_str()); return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::string txt = std::regex_replace(ss.str(), std::regex(R"(<!--[\s\S]*?-->)"), "");
    static const std::regex sp(R"(<species\s+([^>]*)>([\s\S]*?)</species>)");
    static const std::regex ch(R"(<(\w+)\s+([^>]*?)/>)");
    for (std::sregex_iterator it(txt.begin(), txt.end(), sp), end; it != end; ++it) {
        Attrs s = parse_attrs((*it)[1]);
        if (s["id"] != id) continue;
        std::string body = (*it)[2];
        for (std::sregex_iterator jt(body.begin(), body.end(), ch); jt != end; ++jt) {
            if ((*jt)[1] != model) continue;
            out = s;
            for (auto& kv : parse_attrs((*jt)[2])) out[kv.first] = kv.second;
            return true;
        }
        std::fprintf(stderr, "species %s has no <%s> entry in %s\n", id.c_str(), model.c_str(), file.c_str());
        return false;
    }
    std::fprintf(stderr, "species %s not found in %s\n", id.c_str(), file.c_str());
    return false;
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; i++) {
        std::string s(argv[i]);
        auto k = s.find('=');
        if (k != std::string::npos) a[s.substr(0, k)] = s.substr(k + 1);
    }
    auto D = [&](const char* k, double d) { return a.count(k) ? std::stod(a[k]) : d; };
    auto S = [&](const char* k, const char* d) { return a.count(k) ? a[k] : std::string(d); };

    const std::string problem = S("problem", "reflect");
    const double L = D("L", 60.0);
    const int ncell = (int)D("ncell", 750);
    const double dt = D("dt", 0.002);
    const double tend = D("tend", 13.5);
    const double t_sample0 = D("tsample0", 0.0);
    const double dt_sample = D("dtsample", 0.1);
    const double ppc = D("ppc", 50.0);            // particles per cell at n_ref
    const uint64_t seed = (uint64_t)D("seed", 1);
    const std::string out = S("out", "dsmc_out.bin");
    const double sigma = 1.0 / std::sqrt(2.0);
    // Collision model
    const std::string model = S("model", "hs");
    double omega = 0.5, alpha = 1.0;
    std::string species = S("species", "");
    if (model == "vhs" || model == "vss") {
        if (!species.empty()) {
            std::string exe(argv[0]);
            auto k = exe.rfind('/');
            std::string gasfile = S("gas", (k == std::string::npos ? std::string("gas.xml") : exe.substr(0, k + 1) + "gas.xml").c_str());
            Attrs g;
            if (!load_gas(gasfile, species, model, g)) return 1;
            omega = std::stod(g.at("omega"));
            if (model == "vss") alpha = std::stod(g.at("alpha"));
            if (g.count("rotdof") && std::stoi(g["rotdof"]) > 0)
                std::fprintf(stderr, "warning: %s has rotdof=%s; internal energy is ignored (monatomic collisions)\n",
                             species.c_str(), g["rotdof"].c_str());
        }
        omega = D("omega", omega);
        if (model == "vss") alpha = D("alpha", alpha);
    } else if (model != "hs") {
        std::fprintf(stderr, "unknown model\n");
        return 1;
    }
    // sigma_T(g) g = sg_a g^sg_p; HS: sg_a = sigma, sg_p = 1.
    const bool hs_rate = (omega == 0.5);
    const double sg_p = 2.0 - 2.0 * omega;
    const double sg_a = sigma * std::pow(2.0, 2.0 * omega - 1.0) / std::tgamma(2.5 - omega);
    auto sigma_g = [&](double g) { return hs_rate ? sg_a * g : sg_a * std::pow(g, sg_p); };
    const bool isotropic = (alpha == 1.0);
    // mu(T_ref) in lambda units, first Chapman-Enskog approximation (Bird Eqs. 3.75, 4.62)
    const double mu1 = 15.0 * std::sqrt(2.0 * M_PI) * (alpha + 1) * (alpha + 2) / (12.0 * alpha * (7 - 2 * omega) * (5 - 2 * omega));
    std::fprintf(stderr, "model %s%s%s omega %.4g alpha %.4g  mu_ref/(rho sqrt(RT)) = %.6f lambda (1st approx)\n",
                 model.c_str(), species.empty() ? "" : " species ", species.c_str(), omega, alpha, mu1);
    // Problem data
    const double nL = D("nL", 7.0), TL = D("TL", 3.0), nR = D("nR", 1.0), TR = D("TR", 1.0), x0 = D("x0", 0.5 * L);
    const double n1 = D("n1", 1.0), T1 = D("T1", 1.0), U = D("U", 9.585);

    const double dx = L / ncell;
    const double Fn = 1.0 * dx / ppc;  // real particles per simulator at n_ref (unit area)
    Rng rng(seed * 0x9E3779B97F4A7C15ULL + 12345);
    Particles P;

    auto fill = [&](double xa, double xb, double n, double T, double u) {
        double expected = n * (xb - xa) / Fn;
        long N = (long)std::floor(expected + rng.uni());
        double s = std::sqrt(T);
        for (long i = 0; i < N; i++)
            P.add(xa + (xb - xa) * rng.uni(), u + s * rng.nor(), s * rng.nor(), s * rng.nor());
    };
    const bool inflow = (problem == "reflect");
    if (problem == "sod") {
        fill(0.0, x0, nL, TL, 0.0);
        fill(x0, L, nR, TR, 0.0);
    } else if (problem == "reflect") {
        fill(0.0, L, n1, T1, -U);
    } else {
        std::fprintf(stderr, "unknown problem\n");
        return 1;
    }

    std::vector<double> sgmax(ncell, sigma_g(4.0 * std::sqrt(std::max(TL, std::max(T1, TR))) + 2 * U));  // (sigma_T g)_max
    std::vector<double> selx(ncell, 0.0);
    std::vector<int> cnt(ncell), start(ncell), idx;
    std::vector<int> cellof;
    const int nsum = 12;
    FILE* fo = std::fopen(out.c_str(), "wb");
    int32_t hdr[2] = {ncell, nsum};
    std::fwrite(hdr, sizeof(int32_t), 2, fo);
    double hdrd[3] = {L, Fn, dx};
    std::fwrite(hdrd, sizeof(double), 3, fo);

    const long nstep = (long)std::llround(tend / dt);
    double next_sample = t_sample0;
    long ncoll = 0;
    std::vector<double> sums((size_t)ncell * nsum);
    const double coeff = 0.5 * Fn * dt / dx;  // Garcia: 0.5*Fn*(sigma g)_max*dt/Vcell
    for (long step = 0; step <= nstep; step++) {
        double t = step * dt;
        // ---- sample (state at time t, before advancing)
        if (t >= next_sample - 1e-12) {
            std::fill(sums.begin(), sums.end(), 0.0);
            for (size_t i = 0; i < P.size(); i++) {
                int c = std::min(ncell - 1, std::max(0, (int)(P.x[i] / dx)));
                double vx = P.vx[i], vp2 = P.vy[i] * P.vy[i] + P.vz[i] * P.vz[i];
                double* s = &sums[(size_t)c * nsum];
                double v2 = vx * vx;
                s[0] += 1; s[1] += vx; s[2] += v2; s[3] += v2 * vx; s[4] += vp2; s[5] += vx * vp2; s[6] += v2 * v2; s[7] += v2 * vp2;
                s[8] += vp2 * vp2; s[9] += v2 * v2 * vx; s[10] += v2 * vx * vp2; s[11] += vx * vp2 * vp2;
            }
            std::fwrite(&t, sizeof(double), 1, fo);
            std::fwrite(sums.data(), sizeof(double), sums.size(), fo);
            next_sample += dt_sample;
        }
        if (step == nstep) break;
        // ---- inflow buffer at x > L (free-streaming reservoir)
        size_t nold = P.size();
        if (inflow) {
            double W = (std::fabs(U) + 6.0 * std::sqrt(T1)) * dt;
            fill(L, L + W, n1, T1, -U);
        }
        // ---- move + boundaries
        for (size_t i = 0; i < P.size();) {
            P.x[i] += P.vx[i] * dt;
            if (P.x[i] < 0.0) { P.x[i] = -P.x[i]; P.vx[i] = -P.vx[i]; }
            if (P.x[i] >= L) {
                if (inflow) { P.remove(i); if (i < nold) nold--; continue; }
                P.x[i] = 2 * L - P.x[i]; P.vx[i] = -P.vx[i];
                if (P.x[i] >= L) P.x[i] = std::nextafter(L, 0.0);
            }
            i++;
        }
        // ---- sort
        size_t N = P.size();
        std::fill(cnt.begin(), cnt.end(), 0);
        cellof.resize(N);
        for (size_t i = 0; i < N; i++) {
            int c = std::min(ncell - 1, std::max(0, (int)(P.x[i] / dx)));
            cellof[i] = c; cnt[c]++;
        }
        int m = 0;
        for (int c = 0; c < ncell; c++) { start[c] = m; m += cnt[c]; }
        idx.resize(N);
        std::vector<int> fillp(start);
        for (size_t i = 0; i < N; i++) idx[fillp[cellof[i]]++] = (int)i;
        // ---- collide (NTC; HS/VHS rate, isotropic or VSS scattering)
        for (int c = 0; c < ncell; c++) {
            int nc = cnt[c];
            if (nc < 2) continue;
            double sel = coeff * nc * (nc - 1) * sgmax[c] + selx[c];
            long nsel = (long)sel;
            selx[c] = sel - nsel;
            double sgm = sgmax[c];
            for (long k = 0; k < nsel; k++) {
                int p = (int)(rng.uni() * nc); if (p >= nc) p = nc - 1;
                int q = (int)(rng.uni() * (nc - 1)); if (q >= nc - 1) q = nc - 2;
                if (q >= p) q++;
                int i = idx[start[c] + p], j = idx[start[c] + q];
                double gx = P.vx[i] - P.vx[j], gy = P.vy[i] - P.vy[j], gz = P.vz[i] - P.vz[j];
                double g = std::sqrt(gx * gx + gy * gy + gz * gz);
                double sg = sigma_g(g);
                if (sg > sgm) sgm = sg;
                if (sg > sgmax[c] * rng.uni()) {
                    ncoll++;
                    double cx = 0.5 * (P.vx[i] + P.vx[j]), cy = 0.5 * (P.vy[i] + P.vy[j]), cz = 0.5 * (P.vz[i] + P.vz[j]);
                    double hx, hy, hz;  // half post-collision relative velocity
                    if (isotropic) {
                        double ct = 1.0 - 2.0 * rng.uni(), st = std::sqrt(std::max(0.0, 1 - ct * ct)), ph = 2 * M_PI * rng.uni();
                        hx = 0.5 * g * ct; hy = 0.5 * g * st * std::cos(ph); hz = 0.5 * g * st * std::sin(ph);
                    } else {
                        double ct = 2.0 * std::pow(rng.uni(), 1.0 / alpha) - 1.0, st = std::sqrt(std::max(0.0, 1 - ct * ct));
                        double ph = 2 * M_PI * rng.uni(), ce = std::cos(ph), se = std::sin(ph);
                        double gp = std::sqrt(gy * gy + gz * gz);
                        if (gp > 1e-10 * g) {
                            hx = ct * gx + st * se * gp;
                            hy = ct * gy + st * (g * gz * ce - gx * gy * se) / gp;
                            hz = ct * gz - st * (g * gy * ce + gx * gz * se) / gp;
                        } else {
                            hx = ct * gx; hy = st * ce * gx; hz = st * se * gx;
                        }
                        hx *= 0.5; hy *= 0.5; hz *= 0.5;
                    }
                    P.vx[i] = cx + hx; P.vy[i] = cy + hy; P.vz[i] = cz + hz;
                    P.vx[j] = cx - hx; P.vy[j] = cy - hy; P.vz[j] = cz - hz;
                }
            }
            sgmax[c] = sgm;
        }
    }
    std::fclose(fo);
    std::fprintf(stderr, "done: particles %zu collisions %ld\n", P.size(), ncoll);
    return 0;
}
