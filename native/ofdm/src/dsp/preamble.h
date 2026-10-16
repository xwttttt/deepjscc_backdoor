#pragma once
#include "dsp/ofdm_core.h"

namespace ofdm {

// Generate time-domain STF (10 short training symbols, 160 samples)
void generate_stf(cf32* out);

// Generate time-domain LTF (CP + 2 long training symbols, 160 samples)
void generate_ltf(cf32* out);

// Generate time-domain HT-LTF (CP + HT-LTF, 80 samples)
void generate_ht_ltf(cf32* out);

} // namespace ofdm
