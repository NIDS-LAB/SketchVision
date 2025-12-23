#pragma once

namespace sketch {

// Hashing function
unsigned int hash_fn_crc32(const int &adr, const int &prt, const int &sc,
                           const int &seed);
unsigned int hash_fn_mur(const int &adr, const int &prt, const int &sc,
                         const int &seed);
unsigned int hash_fn_xh(const int &adr, const int &prt, const int &sc,
                        const int &seed);
unsigned int hash_fn_fnv1a(const int &adr, const int &prt, const int &sc,
                           const int &seed);
unsigned int hash_fn_jenkins(const int &adr, const int &prt, const int &sc,
                             const int &seed);
unsigned int hash_fn_xor(const int &adr, const int &prt, const int &sc,
                         const int &seed);

unsigned int hash_fn_crc32(const int &sadr, const int &dadr, const int &sprt,
                           const int &dprt, const int &sc, const int &seed);
unsigned int hash_fn_mur(const int &sadr, const int &dadr, const int &sprt,
                         const int &dprt, const int &sc, const int &seed);
unsigned int hash_fn_xh(const int &sadr, const int &dadr, const int &sprt,
                        const int &dprt, const int &sc, const int &seed);
unsigned int hash_fn_fnv1a(const int &sadr, const int &dadr, const int &sprt,
                           const int &dprt, const int &sc, const int &seed);
unsigned int hash_fn_jenkins(const int &sadr, const int &dadr, const int &sprt,
                             const int &dprt, const int &sc, const int &seed);
unsigned int hash_fn_xor(const int &sadr, const int &dadr, const int &sprt,
                         const int &dprt, const int &sc, const int &seed);

} // namespace sketch
