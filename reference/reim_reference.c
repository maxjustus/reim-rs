// Runs the C ReIm pipeline on a WAV file via libsndfile, dumping the output
// waveform and per-frame features for comparison against the Rust port.
#include "reim/analyze_ap.h"
#include "reim/analyze_fo.h"
#include "reim/analyze_silence.h"
#include "reim/analyze_sp.h"
#include "reim/audio_frame.h"
#include "reim/memory.h"
#include "reim/synthesis.h"
#include "reim/vocoder.h"
#include <sndfile.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s in.wav out.wav feat.csv sp_ap.csv\n", argv[0]);
        return 1;
    }
    SF_INFO in_info = { 0 };
    SNDFILE* fin = sf_open(argv[1], SFM_READ, &in_info);
    if (!fin) { fprintf(stderr, "open in: %s\n", sf_strerror(NULL)); return 1; }
    const double fs = in_info.samplerate;

    SF_INFO out_info = { 0 };
    out_info.channels = 1;
    out_info.samplerate = in_info.samplerate;
    out_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* fout = sf_open(argv[2], SFM_WRITE, &out_info);
    FILE* feat = fopen(argv[3], "w");
    FILE* spap = fopen(argv[4], "w");

    const double period = 5.0, fo_floor = 71.0, fo_ceil = 800.0;
    const size_t fftsize = 2048, numbins = fftsize / 2 + 1;

    audio_frame_t* afr = create_audio_frame(fs, period, fftsize);
    vocoder_context_t* voc = create_vocoder_context(period, fftsize, fo_floor, fo_ceil, fs);
    fo_context_t* foc = create_fo_context(voc);
    ap_context_t* apc = create_ap_context(voc);
    sp_context_t* spc = create_sp_context(voc);
    synthesis_context_t* syn = create_synthesis_context(voc);

    double* wf = allocate_vector(fftsize + 1);
    double* ap = allocate_vector(numbins);
    double* sp = allocate_vector(numbins);
    const double* wave = wf + 1;
    const double* wave_d = wf;

    double x;
    long n = 0, frame = 0;
    while (sf_read_double(fin, &x, 1) == 1) {
        if (next_audio_frame(afr, x, wf)) {
            bool sil = analyze_silence(voc, wave, REIM_SILENCE_THRESHOLD);
            double fo = analyze_fo(voc, foc, wave, wave_d);
            bool vu = analyze_ap(voc, apc, wave, fo, sil, ap);
            analyze_sp(voc, spc, wave, fo, vu, sil, sp);
            synthesize_new_frame(voc, syn, fo, vu, sil, ap, sp);
            fprintf(feat, "%ld,%d,%.17g,%d\n", frame, sil ? 1 : 0, fo, vu ? 1 : 0);
            // dump sp/ap for a handful of frames to keep file small
            if (frame % 50 == 0) {
                for (size_t k = 0; k < numbins; k++)
                    fprintf(spap, "%ld,%zu,%.17g,%.17g\n", frame, k, sp[k], ap[k]);
            }
            frame++;
        }
        double y = synthesize_next_sample(voc, syn);
        sf_write_double(fout, &y, 1);
        n++;
    }
    printf("samples=%ld frames=%ld fs=%.0f\n", n, frame, fs);

    fclose(feat); fclose(spap); sf_close(fin); sf_close(fout);
    return 0;
}
