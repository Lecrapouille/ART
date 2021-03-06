# Adaptive Radix Tree

This repo contains a modified implementation of the Adaptive Radix Tree (ART) as described in:

```
The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases",
Viktor Leis, Alfons Kemper, and Thomas Neumann
ICDE 2013
```

See original version:
- [Publication](https://db.in.tum.de/~leis/papers/ART.pdf)
- [Code source](https://db.in.tum.de/~leis/index/ART.tgz)

## Notes:

- Two versions are included, one with path compression (ART.cpp), and one
  without (ARTshort.cpp).

- Both version are meant as secondary index structures. Because of lazy
  expansion, it must be possible to retrieve the key from the database (using
  the loadKey function).

- ART maps a key of arbitrary length to a tuple identifier with the same size as
  pointers. The most significant bit of the tuple identifier is always cleared
  when retrieving it from the tree in order to distinguish pointers from leaves
  (pointer tagging).

- This implementation assumes that all keys stored in a tree are distinct. The
  insert function can easily be modified to detect if a duplicate key is
  inserted.

- The code was tested using g++ and clang++ on 64-bit linux.
