#include "hash.hpp"
#include <sstream>

namespace sketch {
uint32_t murmur_hash_3(const std::string &key, uint32_t seed) {
  auto *data = (const uint8_t *)key.data();
  const size_t len = key.size();
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;
  const uint32_t r1 = 15;
  const uint32_t r2 = 13;
  const uint32_t m = 5;
  const uint32_t n = 0xe6546b64;
  uint32_t hash = seed;

  size_t i;
  for (i = 0; i <= len - 4; i += 4) {
    uint32_t k = *(uint32_t *)(data + i);
    k *= c1;
    k = (k << r1) | (k >> (32 - r1));
    k *= c2;
    hash ^= k;
    hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
  }

  uint32_t k = 0;
  for (size_t j = i; j < len; ++j) {
    k ^= data[j];
    k *= c1;
    k = (k << r1) | (k >> (32 - r1));
    k *= c2;
    hash ^= k;
    hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
  }

  hash ^= len;
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);

  return hash;
}

uint32_t crc32(const std::string &data) {
  uint32_t crc = 0xFFFFFFFF;
  for (size_t i = 0; i < data.size(); ++i) {
    uint8_t byte = data[i];
    crc = crc ^ byte;
    for (uint8_t j = 0; j < 8; j++) {
      crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
  }
  return ~crc;
}

unsigned int hash_fn_crc32(const int &adr, const int &prt, const int &sc,
                           const int &seed) {
  std::stringstream ss;
  ss << adr << prt << sc << seed; // Including seed in the key
  std::string key = ss.str();

  return crc32(key);
}

unsigned int hash_fn_mur(const int &adr, const int &prt, const int &sc,
                         const int &seed) {
  std::stringstream ss;
  ss << adr << prt << sc;
  std::string key = ss.str();

  return murmur_hash_3(key, seed);
}

unsigned int hash_fn_fnv1a(const int &adr, const int &prt, const int &sc,
                           const int &seed) {
  std::stringstream ss;
  ss << adr << prt << sc << seed; // Including seed in the key
  std::string key = ss.str();

  const unsigned int prime = 0x01000193; // 16777619
  unsigned int hash = 0x811C9DC5;        // 2166136261
  for (size_t i = 0; i < key.length(); i++) {
    hash ^= key[i];
    hash *= prime;
  }
  return hash;
}

unsigned int hash_fn_jenkins(const int &adr, const int &prt, const int &sc,
                             const int &seed) {
  std::stringstream ss;
  ss << adr << prt << sc << seed; // Including seed in the key
  std::string key = ss.str();

  unsigned int hash = 0;
  for (size_t i = 0; i < key.length(); i++) {
    hash += key[i];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

unsigned int hash_fn_xor(const int &adr, const int &prt, const int &sc,
                         const int &seed) {
  std::stringstream ss;
  ss << adr << prt << sc << seed; // Including seed in the key
  std::string key = ss.str();

  unsigned int hash = 0;
  for (size_t i = 0; i < key.length(); i++) {
    hash ^= (key[i] + 0x9e3779b9 + (hash << 6) + (hash >> 2));
  }
  return hash;
}

unsigned int hash_fn_crc32(const int &sadr, const int &dadr, const int &sprt,
                           const int &dprt, const int &sc, const int &seed) {
  std::stringstream ss;
  ss << sadr << dadr << sprt << dprt << sc << seed;
  std::string key = ss.str();

  return crc32(key);
}

unsigned int hash_fn_mur(const int &sadr, const int &dadr, const int &sprt,
                         const int &dprt, const int &sc, const int &seed) {
  std::stringstream ss;
  ss << sadr << dadr << sprt << dprt << sc;
  std::string key = ss.str();

  return murmur_hash_3(key, seed);
}

unsigned int hash_fn_fnv1a(const int &sadr, const int &dadr, const int &sprt,
                           const int &dprt, const int &sc, const int &seed) {
  std::stringstream ss;
  ss << sadr << dadr << sprt << dprt << sc << seed;
  std::string key = ss.str();

  const unsigned int prime = 0x01000193; // 16777619
  unsigned int hash = 0x811C9DC5;        // 2166136261
  for (size_t i = 0; i < key.length(); i++) {
    hash ^= key[i];
    hash *= prime;
  }
  return hash;
}

unsigned int hash_fn_jenkins(const int &sadr, const int &dadr, const int &sprt,
                             const int &dprt, const int &sc, const int &seed) {
  std::stringstream ss;
  ss << sadr << dadr << sprt << dprt << sc << seed;
  std::string key = ss.str();

  unsigned int hash = 0;
  for (size_t i = 0; i < key.length(); i++) {
    hash += key[i];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

unsigned int hash_fn_xor(const int &sadr, const int &dadr, const int &sprt,
                         const int &dprt, const int &sc, const int &seed) {
  std::stringstream ss;
  ss << sadr << dadr << sprt << dprt << sc << seed;
  std::string key = ss.str();

  unsigned int hash = 0;
  for (size_t i = 0; i < key.length(); i++) {
    hash ^= (key[i] + 0x9e3779b9 + (hash << 6) + (hash >> 2));
  }
  return hash;
}

} // namespace sketch
