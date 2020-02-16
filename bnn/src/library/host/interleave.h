#pragma once

// Interleave x and y. The bits of x will be in the even
// positions, and the bits of y in the odd positions.
inline uint16_t interleave(uint8_t x_in, uint8_t y_in) {
    uint16_t x = x_in;
    uint16_t y = y_in;
    uint16_t z;

    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    x = (x | (x << 1)) & 0x5555;

    y = (y | (y << 4)) & 0x0F0F;
    y = (y | (y << 2)) & 0x3333;
    y = (y | (y << 1)) & 0x5555;

    z = x | (y << 1);

    return z;
}


// Interleave x and y. The bits of x will be in the even
// positions, and the bits of y in the odd positions.
inline uint32_t interleave(uint16_t x_in, uint16_t y_in) {
    uint32_t x = x_in;
    uint32_t y = y_in;
    uint32_t z;

    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    z = x | (y << 1);

    return z;
}


// Interleave x and y. The bits of x will be in the even
// positions, and the bits of y in the odd positions.
inline uint64_t interleave(uint32_t x_in, uint32_t y_in) {
    uint64_t x = x_in;
    uint64_t y = y_in;
    uint64_t z;

    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FF;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2))  & 0x3333333333333333;
    x = (x | (x << 1))  & 0x5555555555555555;

    y = (y | (y << 16)) & 0x0000FFFF0000FFFF;
    y = (y | (y << 8))  & 0x00FF00FF00FF00FF;
    y = (y | (y << 4))  & 0x0F0F0F0F0F0F0F0F;
    y = (y | (y << 2))  & 0x3333333333333333;
    y = (y | (y << 1))  & 0x5555555555555555;

    z = x | (y << 1);

    return z;
}