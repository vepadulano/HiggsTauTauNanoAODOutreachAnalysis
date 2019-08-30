#ifndef SKIM_H
#define SKIM_H

namespace Helper {
template <typename T>
float DeltaPhi(T v1, T v2, const T c = M_PI)
{
    auto r = std::fmod(v2 - v1, 2.0 * c);
    if (r < -c) {
        r += 2.0 * c;
    }
    else if (r > c) {
        r -= 2.0 * c;
    }
    return r;
}
}

#endif
