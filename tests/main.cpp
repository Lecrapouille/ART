#include "ART.cpp"
#include <sys/time.h>  // gettime
#include <algorithm>   // std::random_shuffle
#include <stdio.h>
#include <cassert>

// -----------------------------------------------------------------------------
static double gettime(void)
{
  struct timeval now_tv;
  gettimeofday (&now_tv, nullptr);
  return ((double)now_tv.tv_sec) + ((double)now_tv.tv_usec) / 1000000.0;
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  if (argc != 3)
    {
      printf("usage: %s n 0|1|2\nn: number of keys\n0: sorted keys\n1: dense keys\n2: sparse keys\n", argv[0]);
      return 1;
    }

  uint64_t n = atoi(argv[1]);
  uint64_t* keys = new uint64_t[n];

  // Generate keys
  for (uint64_t i = 0; i < n; ++i)
    {
      // dense, sorted
      keys[i] = i+1;
    }
  if (atoi(argv[2]) == 1)
    {
      // dense, random
      std::random_shuffle(keys, keys+n);
    }
  if (atoi(argv[2]) == 2)
    {
      // "pseudo - sparse" (the most - significant leaf bit gets lost)
      for (uint64_t i = 0; i < n; ++i)
        {
          keys[i] = (static_cast<uint64_t>(rand()) << 32) | static_cast<uint64_t>(rand());
        }
    }

  // Build tree
  double start = gettime();
  art::Node* tree = nullptr;
  for (uint64_t i = 0; i < n; ++i)
    {
      uint8_t key[8]; art::loadKey(keys[i], key);
      art::insert(tree, &tree, key, 0, keys[i], 8);
    }
  printf("insert, %ld, %f\n", n, (n / 1000000.0) / (gettime() - start));

  // Repeat lookup for small trees to get reproducable results
  uint64_t repeat = 10000000 / n;
  if (repeat < 1)
    repeat = 1;
  start = gettime();
  for (uint64_t r = 0; r < repeat; ++r)
    {
      for (uint64_t i = 0; i < n; ++i)
        {
          uint8_t key[8]; art::loadKey(keys[i], key);
            art::Node* leaf = art::lookup(tree, key, 8, 0, 8);
          assert(art::isLeaf(leaf) && art::getLeafValue(leaf) == keys[i]);
        }
    }
  printf("lookup, %ld, %f\n", n, (n*repeat / 1000000.0) / (gettime() - start));

  start = gettime();
  for (uint64_t i = 0; i < n; ++i)
    {
      uint8_t key[8]; art::loadKey(keys[i], key);
      art::erase(tree, &tree, key, 8, 0, 8);
    }
  printf("erase, %ld, %f\n", n, (n / 1000000.0) / (gettime() - start));
  assert(tree == nullptr);

  return 0;
}
