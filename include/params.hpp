#pragma once
struct RtParams { unsigned sample_rate=16000, block=128, in_ch=2, out_ch=1; };
struct FxParams { unsigned L=128, Ls=128; float mu=5e-4f, leak=1e-4f, eps=1e-6f; };
