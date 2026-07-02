// WORLD D4C aperiodicity reference (for validating reim's aperiodicity port).
// Runs WORLD's D4C on a waveform + an externally-supplied F0 contour, so the
// comparison isolates the aperiodicity computation from F0-estimation differences.
//
// Build (needs a checkout of github.com/mmorise/World):
//   git clone --depth 1 https://github.com/mmorise/World /tmp/World
//   (cd /tmp/World && c++ -O2 -std=c++11 -Isrc -c src/*.cpp && ar rcs libworld.a *.o)
//   c++ -O2 -std=c++11 -I/tmp/World/src reference/d4c_reference.cpp /tmp/World/libworld.a -o /tmp/d4c_reference
// Feed it raw-f64 audio + reim's "time,f0" CSV (from `reim f0`); see eval/ notes.
// args: x.f64  f0.csv("time,f0")  fs  fft_size  out.f64
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "world/d4c.h"

int main(int argc, char **argv) {
  if (argc != 6) { fprintf(stderr, "args: x.f64 f0.csv fs fft_size out.f64\n"); return 2; }
  int fs = atoi(argv[3]);
  int fft_size = atoi(argv[4]);

  FILE *fx = fopen(argv[1], "rb");
  fseek(fx, 0, SEEK_END); long bytes = ftell(fx); fseek(fx, 0, SEEK_SET);
  int x_length = (int)(bytes / 8);
  std::vector<double> x(x_length);
  if (fread(x.data(), 8, x_length, fx) != (size_t)x_length) { fprintf(stderr, "read x\n"); return 1; }
  fclose(fx);

  std::vector<double> tpos, f0v;
  FILE *ff = fopen(argv[2], "r");
  double t, f;
  while (fscanf(ff, "%lf,%lf", &t, &f) == 2) { tpos.push_back(t); f0v.push_back(f); }
  fclose(ff);
  int f0_length = (int)tpos.size();

  int numbins = fft_size / 2 + 1;
  double **aper = (double **)malloc(f0_length * sizeof(double *));
  for (int i = 0; i < f0_length; i++) aper[i] = (double *)malloc(numbins * sizeof(double));

  D4COption opt;
  InitializeD4COption(&opt);
  D4C(x.data(), x_length, fs, tpos.data(), f0v.data(), f0_length, fft_size, &opt, aper);

  FILE *fo = fopen(argv[5], "wb");
  for (int i = 0; i < f0_length; i++) fwrite(aper[i], 8, numbins, fo);
  fclose(fo);
  printf("D4C: %d frames x %d bins (fs=%d fft_size=%d, threshold=%.3f)\n", f0_length, numbins, fs, fft_size, opt.threshold);
  return 0;
}
