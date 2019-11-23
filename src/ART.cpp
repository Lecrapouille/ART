// Adaptive Radix Tree
// Viktor Leis, 2012 leis@in.tum.de

#include "ART.hpp"
#include <cstring>     // memset, memcpy
#include <emmintrin.h> // x86 SSE intrinsics
#include <algorithm> // std::min

namespace art {

// -----------------------------------------------------------------------------
//! \brief This address is used to communicate that search failed.
// -----------------------------------------------------------------------------
static Node* nullNode = nullptr;

// -----------------------------------------------------------------------------
//! \brief Create a pseudo leaf.
// -----------------------------------------------------------------------------
static inline Node* makeLeaf(uintptr_t tid)
{
  return reinterpret_cast<Node*>((tid << 1) | 1);
}

// -----------------------------------------------------------------------------
//! \brief The the value stored in the pseudo leaf.
// -----------------------------------------------------------------------------
static inline uintptr_t getLeafValue(Node* node)
{
  return reinterpret_cast<uintptr_t>(node) >> 1;
}

// -----------------------------------------------------------------------------
//! \brief Is the node a leaf?
// -----------------------------------------------------------------------------
static inline bool isLeaf(Node* node)
{
  return reinterpret_cast<uintptr_t>(node) & 1;
}

// -----------------------------------------------------------------------------
//! \brief Flip the sign bit, enables signed SSE comparison of uint32_t values,
//! used by Node16.
// -----------------------------------------------------------------------------
static uint8_t flipSign(uint8_t keyByte)
{
  return keyByte ^ 128;
}

// -----------------------------------------------------------------------------
//! \brief Store the key of the tuple into the key vector.
//! Implementation is database specific
// -----------------------------------------------------------------------------
static void loadKey(uintptr_t tid, uint8_t key[])
{
  reinterpret_cast<uint64_t*>(key)[0] = __builtin_bswap64(tid);
}

// -----------------------------------------------------------------------------
//! \brief
// -----------------------------------------------------------------------------
static inline uint32_t ctz(uint16_t x)
{
  // Count trailing zeros, only defined for x>0
#ifdef __GNUC__
  return __builtin_ctz(x);
#else
  // Adapted from Hacker's Delight
  uint32_t n = 1;
  if ((x & 0xFF)  == 0) { n += 8; x = x >> 8; }
  if ((x & 0x0F)  == 0) { n += 4; x = x >> 4; }
  if ((x & 0x03)  == 0) { n += 2; x = x >> 2; }
  return n - (x & 1);
#endif
}

// -----------------------------------------------------------------------------
//! \brief Find the next child for the keyByte.
// -----------------------------------------------------------------------------
static Node** findChild(Node* n, uint8_t keyByte)
{
  switch (n->type)
    {
    case NodeType4:
      {
        Node4* node = static_cast<Node4*>(n);
        for (uint32_t i = 0; i < node->count; ++i)
          if (node->key[i] == keyByte)
            return &node->child[i];
        return &nullNode;
      }
    case NodeType16:
      {
        Node16* node = static_cast<Node16*>(n);
        __m128i cmp = _mm_cmpeq_epi8(_mm_set1_epi8(flipSign(keyByte)),
                                     _mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
        uint32_t bitfield = _mm_movemask_epi8(cmp)&((1 << node->count) - 1);
        if (bitfield)
          return &node->child[ctz(bitfield)];
        else
          return &nullNode;
      }
    case NodeType48:
      {
        Node48* node = static_cast<Node48*>(n);
        if (node->childIndex[keyByte] != emptyMarker)
          return &node->child[node->childIndex[keyByte]];
        else
          return &nullNode;
      }
    case NodeType256:
      {
        Node256* node = static_cast<Node256*>(n);
        return &(node->child[keyByte]);
      }
    }
  throw; // Unreachable
}

// -----------------------------------------------------------------------------
//! \brief Find the leaf with smallest key
// -----------------------------------------------------------------------------
static Node* minimum(Node* node)
{
  if (!node)
    return nullptr;

  if (isLeaf(node))
    return node;

  switch (node->type)
    {
    case NodeType4:
      {
        Node4* n = static_cast<Node4*>(node);
        return minimum(n->child[0]);
      }
    case NodeType16:
      {
        Node16* n = static_cast<Node16*>(node);
        return minimum(n->child[0]);
      }
    case NodeType48:
      {
        Node48* n = static_cast<Node48*>(node);
        uint32_t pos = 0;
        while (n->childIndex[pos] == emptyMarker)
          ++pos;
        return minimum(n->child[n->childIndex[pos]]);
      }
    case NodeType256:
      {
        Node256* n = static_cast<Node256*>(node);
        uint32_t pos = 0;
        while (!n->child[pos])
          ++pos;
        return minimum(n->child[pos]);
      }
    }
  throw; // Unreachable
}

// -----------------------------------------------------------------------------
//! \brief Find the leaf with largest key
// -----------------------------------------------------------------------------
static Node* maximum(Node* node)
{
  if (!node)
    return nullptr;

  if (isLeaf(node))
    return node;

  switch (node->type)
    {
    case NodeType4:
      {
        Node4* n = static_cast<Node4*>(node);
        return maximum(n->child[n->count - 1]);
      }
    case NodeType16:
      {
        Node16* n = static_cast<Node16*>(node);
        return maximum(n->child[n->count - 1]);
      }
    case NodeType48:
      {
        Node48* n = static_cast<Node48*>(node);
        uint32_t pos = 255;
        while (n->childIndex[pos] == emptyMarker)
          pos--;
        return maximum(n->child[n->childIndex[pos]]);
      }
    case NodeType256:
      {
        Node256* n = static_cast<Node256*>(node);
        uint32_t pos = 255;
        while (!n->child[pos])
          pos--;
        return maximum(n->child[pos]);
      }
    }
  throw; // Unreachable
}

// -----------------------------------------------------------------------------
//! \brief Check if the key of the leaf is equal to the searched key
// -----------------------------------------------------------------------------
static bool leafMatches(Node* leaf, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength)
{
  if (depth != keyLength)
    {
      uint8_t leafKey[maxKeyLength];
      loadKey(getLeafValue(leaf), leafKey);
      for (uint32_t i = depth; i < keyLength; ++i)
        if (leafKey[i] != key[i])
          return false;
    }
  return true;
}

// -----------------------------------------------------------------------------
//! \brief Compare the key with the prefix of the node, return the number matching bytes
// -----------------------------------------------------------------------------
static uint32_t prefixMismatch(Node* node, uint8_t key[], uint32_t depth, uint32_t maxKeyLength)
{
  uint32_t pos;
  if (node->prefixLength>maxPrefixLength)
    {
      for (pos = 0; pos < maxPrefixLength; ++pos)
        if (key[depth + pos] != node->prefix[pos])
          return pos;
      uint8_t minKey[maxKeyLength];
      loadKey(getLeafValue(minimum(node)), minKey);
      for (; pos < node->prefixLength; ++pos)
        if (key[depth + pos] != minKey[depth + pos])
          return pos;
    }
  else
    {
      for (pos = 0; pos < node->prefixLength; ++pos)
        if (key[depth + pos] != node->prefix[pos])
          return pos;
    }
  return pos;
}

// -----------------------------------------------------------------------------
//! \brief Find the node with a matching key, optimistic version
// -----------------------------------------------------------------------------
/*static*/ Node* lookup(Node* node, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength)
{
  bool skippedPrefix = false; // Did we optimistically skip some prefix without checking it?

  while (node != nullptr)
    {
      if (isLeaf(node))
        {
          if (!skippedPrefix && depth == keyLength) // No check required
            return node;

          if (depth != keyLength)
            {
              // Check leaf
              uint8_t leafKey[maxKeyLength];
              loadKey(getLeafValue(node), leafKey);
              for (uint32_t i = (skippedPrefix ? 0 : depth); i < keyLength; ++i)
                if (leafKey[i] != key[i])
                  return nullptr;
            }
          return node;
        }

      if (node->prefixLength)
        {
          if (node->prefixLength < maxPrefixLength)
            {
              for (uint32_t pos = 0; pos < node->prefixLength; ++pos)
                if (key[depth + pos] != node->prefix[pos])
                  return nullptr;
            } else
            skippedPrefix = true;
          depth += node->prefixLength;
        }

      node = *findChild(node, key[depth]);
      depth++;
    }

  return nullptr;
}

// -----------------------------------------------------------------------------
//! \brief Find the node with a matching key, optimistic version
// -----------------------------------------------------------------------------
//Node* lookup(Node* node, uint8_t key[], uint32_t keyLength)
//{
//  return lookup(node, key, keyLength, 0u, maxPrefixLength - 1u);
//}

// -----------------------------------------------------------------------------
//! \brief Find the node with a matching key, alternative pessimistic version
// -----------------------------------------------------------------------------
Node* lookupPessimistic(Node* node, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength)
{
  while (node != nullptr)
    {
      if (isLeaf(node))
        {
          if (leafMatches(node, key, keyLength, depth, maxKeyLength))
            return node;
          return nullptr;
        }

      if (prefixMismatch(node, key, depth, maxKeyLength) != node->prefixLength)
        return nullptr;
      else
        depth += node->prefixLength;

      node = *findChild(node, key[depth]);
      depth++;
    }

  return nullptr;
}

// -----------------------------------------------------------------------------
//! \brief Helper function that copies the prefix from the source to the destination node
// -----------------------------------------------------------------------------
static void copyPrefix(Node* src, Node* dst)
{
  dst->prefixLength = src->prefixLength;
  memcpy(dst->prefix, src->prefix, std::min(src->prefixLength, maxPrefixLength));
}

// -----------------------------------------------------------------------------
//! \brief Insert leaf into inner node
// -----------------------------------------------------------------------------
static void insertNode256(Node256* node, Node** /*nodeRef*/, uint8_t keyByte, Node* child)
{
  node->count++;
  node->child[keyByte] = child;
}

// -----------------------------------------------------------------------------
//! \brief Insert leaf into inner node
// -----------------------------------------------------------------------------
static void insertNode48(Node48* node, Node** nodeRef, uint8_t keyByte, Node* child)
{
  if (node->count < 48)
    {
      // Insert element
      uint32_t pos = node->count;
      if (node->child[pos])
        for (pos = 0; node->child[pos] != nullptr; ++pos);
      node->child[pos] = child;
      node->childIndex[keyByte] = pos;
      node->count++;
    }
  else
    {
      // Grow to Node256
      Node256* newNode = new Node256();
      for (uint32_t i = 0; i < 256; ++i)
        if (node->childIndex[i] != 48)
          newNode->child[i] = node->child[node->childIndex[i]];
      newNode->count = node->count;
      copyPrefix(node, newNode);
      *nodeRef = newNode;
      delete node;
      return insertNode256(newNode, nodeRef, keyByte, child);
    }
}

// -----------------------------------------------------------------------------
//! \brief Insert leaf into inner node
// -----------------------------------------------------------------------------
static void insertNode16(Node16* node, Node** nodeRef, uint8_t keyByte, Node* child)
{
  if (node->count < 16)
    {
      // Insert element
      uint8_t keyByteFlipped = flipSign(keyByte);
      __m128i cmp = _mm_cmplt_epi8(_mm_set1_epi8(keyByteFlipped), _mm_loadu_si128(reinterpret_cast<__m128i*>(node->key)));
      uint16_t bitfield = _mm_movemask_epi8(cmp)&(0xFFFF >> (16 - node->count));
      uint32_t pos = bitfield?ctz(bitfield):node->count;
      memmove(node->key + pos + 1, node->key + pos, node->count - pos);
      memmove(node->child + pos + 1, node->child + pos, (node->count - pos) * sizeof(uintptr_t));
      node->key[pos] = keyByteFlipped;
      node->child[pos] = child;
      node->count++;
    }
  else
    {
      // Grow to Node48
      Node48* newNode = new Node48();
      *nodeRef = newNode;
      memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
      for (uint32_t i = 0; i < node->count; ++i)
        newNode->childIndex[flipSign(node->key[i])] = i;
      copyPrefix(node, newNode);
      newNode->count = node->count;
      delete node;
      return insertNode48(newNode, nodeRef, keyByte, child);
    }
}

// -----------------------------------------------------------------------------
//! \brief Insert leaf into inner node
// -----------------------------------------------------------------------------
static void insertNode4(Node4* node, Node** nodeRef, uint8_t keyByte, Node* child)
{
  if (node->count < 4)
    {
      // Insert element
      uint32_t pos;
      for (pos = 0; (pos < node->count) && (node->key[pos] < keyByte); ++pos);
      memmove(node->key + pos + 1, node->key + pos, node->count - pos);
      memmove(node->child + pos + 1, node->child + pos, (node->count - pos) * sizeof(uintptr_t));
      node->key[pos] = keyByte;
      node->child[pos] = child;
      node->count++;
    }
  else
    {
      // Grow to Node16
      Node16* newNode = new Node16();
      *nodeRef = newNode;
      newNode->count = 4;
      copyPrefix(node, newNode);
      for (uint32_t i = 0; i < 4; ++i)
        newNode->key[i] = flipSign(node->key[i]);
      memcpy(newNode->child, node->child, node->count * sizeof(uintptr_t));
      delete node;
      return insertNode16(newNode, nodeRef, keyByte, child);
    }
}

// -----------------------------------------------------------------------------
//! \brief Insert the leaf value into the tree
// -----------------------------------------------------------------------------
/*static*/ void insert(Node* node, Node** nodeRef, uint8_t key[], uint32_t depth, uintptr_t value, uint32_t maxKeyLength)
{
  if (node == nullptr)
    {
      *nodeRef = makeLeaf(value);
      return;
    }

  if (isLeaf(node))
    {
      // Replace leaf with Node4 and store both leaves in it
      uint8_t existingKey[maxKeyLength];
      loadKey(getLeafValue(node), existingKey);
      uint32_t newPrefixLength = 0;
      while (existingKey[depth + newPrefixLength] == key[depth + newPrefixLength])
        newPrefixLength++;

      Node4* newNode = new Node4();
      newNode->prefixLength = newPrefixLength;
      memcpy(newNode->prefix, key + depth, std::min(newPrefixLength, maxPrefixLength));
      *nodeRef = newNode;

      insertNode4(newNode, nodeRef, existingKey[depth + newPrefixLength], node);
      insertNode4(newNode, nodeRef, key[depth + newPrefixLength], makeLeaf(value));
      return;
    }

  // Handle prefix of inner node
  if (node->prefixLength)
    {
      uint32_t mismatchPos = prefixMismatch(node, key, depth, maxKeyLength);
      if (mismatchPos != node->prefixLength)
        {
          // Prefix differs, create new node
          Node4* newNode = new Node4();
          *nodeRef = newNode;
          newNode->prefixLength = mismatchPos;
          memcpy(newNode->prefix, node->prefix, std::min(mismatchPos, maxPrefixLength));
          // Break up prefix
          if (node->prefixLength < maxPrefixLength)
            {
              insertNode4(newNode, nodeRef, node->prefix[mismatchPos], node);
              node->prefixLength -= (mismatchPos + 1);
              memmove(node->prefix, node->prefix + mismatchPos + 1, std::min(node->prefixLength, maxPrefixLength));
            }
          else
            {
              node->prefixLength -= (mismatchPos + 1);
              uint8_t minKey[maxKeyLength];
              loadKey(getLeafValue(minimum(node)), minKey);
              insertNode4(newNode, nodeRef, minKey[depth + mismatchPos], node);
              memmove(node->prefix, minKey + depth + mismatchPos + 1, std::min(node->prefixLength, maxPrefixLength));
            }
          insertNode4(newNode, nodeRef, key[depth + mismatchPos], makeLeaf(value));
          return;
        }
      depth += node->prefixLength;
    }

  // Recurse
  Node** child = findChild(node, key[depth]);
  if (*child)
    {
      insert(*child, child, key, depth + 1, value, maxKeyLength);
      return;
    }

  // Insert leaf into inner node
  Node* newNode = makeLeaf(value);
  switch (node->type)
    {
    case NodeType4: insertNode4(static_cast<Node4*>(node), nodeRef, key[depth], newNode); break;
    case NodeType16: insertNode16(static_cast<Node16*>(node), nodeRef, key[depth], newNode); break;
    case NodeType48: insertNode48(static_cast<Node48*>(node), nodeRef, key[depth], newNode); break;
    case NodeType256: insertNode256(static_cast<Node256*>(node), nodeRef, key[depth], newNode); break;
    }
}

// -----------------------------------------------------------------------------
//! \brief Insert the leaf value into the tree
// -----------------------------------------------------------------------------
//static void insert(Node* node, uint8_t key[], uintptr_t value)
//{
//  insert(node, &node, key, 0u, maxPrefixLength - 1u);
//}

// -----------------------------------------------------------------------------
//! \brief Delete leaf from inner node
// -----------------------------------------------------------------------------
static void eraseNode4(Node4* node, Node** nodeRef, Node** leafPlace)
{
  uint32_t pos = leafPlace - node->child;
  memmove(node->key + pos, node->key + pos + 1, node->count - pos - 1);
  memmove(node->child + pos, node->child + pos + 1, (node->count - pos - 1) * sizeof(uintptr_t));
  node->count--;

  if (node->count == 1)
    {
      // Get rid of one - way node
      Node* child = node->child[0];
      if (!isLeaf(child))
        {
          // Concantenate prefixes
          uint32_t l1 = node->prefixLength;
          if (l1 < maxPrefixLength)
            {
              node->prefix[l1] = node->key[0];
              l1++;
            }
          if (l1 < maxPrefixLength)
            {
              uint32_t l2 = std::min(child->prefixLength, maxPrefixLength - l1);
              memcpy(node->prefix + l1, child->prefix, l2);
              l1 += l2;
            }
          // Store concantenated prefix
          memcpy(child->prefix, node->prefix, std::min(l1, maxPrefixLength));
          child->prefixLength += node->prefixLength + 1;
        }
      *nodeRef = child;
      delete node;
    }
}

// -----------------------------------------------------------------------------
//! \brief Delete leaf from inner node
// -----------------------------------------------------------------------------
static void eraseNode16(Node16* node, Node** nodeRef, Node** leafPlace)
{
  uint32_t pos = leafPlace - node->child;
  memmove(node->key + pos, node->key + pos + 1, node->count - pos - 1);
  memmove(node->child + pos, node->child + pos + 1, (node->count - pos - 1) * sizeof(uintptr_t));
  node->count--;

  if (node->count == 3)
    {
      // Shrink to Node4
      Node4* newNode = new Node4();
      newNode->count = node->count;
      copyPrefix(node, newNode);
      for (uint32_t i = 0; i < 4; ++i)
        newNode->key[i] = flipSign(node->key[i]);
      memcpy(newNode->child, node->child, sizeof(uintptr_t)*4);
      *nodeRef = newNode;
      delete node;
    }
}

// -----------------------------------------------------------------------------
//! \brief Delete leaf from inner node
// -----------------------------------------------------------------------------
static void eraseNode48(Node48* node, Node** nodeRef, uint8_t keyByte)
{
  node->child[node->childIndex[keyByte]] = nullptr;
  node->childIndex[keyByte] = emptyMarker;
  node->count--;

  if (node->count == 12)
    {
      // Shrink to Node16
      Node16 *newNode = new Node16();
      *nodeRef = newNode;
      copyPrefix(node, newNode);
      for (uint32_t b = 0; b < 256; b++)
        {
          if (node->childIndex[b] != emptyMarker)
            {
              newNode->key[newNode->count] = flipSign(b);
              newNode->child[newNode->count] = node->child[node->childIndex[b]];
              newNode->count++;
            }
        }
      delete node;
    }
}

// -----------------------------------------------------------------------------
//! \brief Delete leaf from inner node
// -----------------------------------------------------------------------------
static void eraseNode256(Node256* node, Node** nodeRef, uint8_t keyByte)
{
  node->child[keyByte] = nullptr;
  node->count--;

  if (node->count == 37)
    {
      // Shrink to Node48
      Node48 *newNode = new Node48();
      *nodeRef = newNode;
      copyPrefix(node, newNode);
      for (uint32_t b = 0; b < 256; b++)
        {
          if (node->child[b])
            {
              newNode->childIndex[b] = newNode->count;
              newNode->child[newNode->count] = node->child[b];
              newNode->count++;
            }
        }
      delete node;
    }
}

// -----------------------------------------------------------------------------
//! \brief
// -----------------------------------------------------------------------------
void erase(Node* node, Node** nodeRef, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength)
{
  if (!node)
    return;

  if (isLeaf(node))
    {
      // Make sure we have the right leaf
      if (leafMatches(node, key, keyLength, depth, maxKeyLength))
        *nodeRef = nullptr;
      return;
    }

  // Handle prefix
  if (node->prefixLength)
    {
      if (prefixMismatch(node, key, depth, maxKeyLength) != node->prefixLength)
        return;
      depth += node->prefixLength;
    }

  Node** child = findChild(node, key[depth]);
  if (isLeaf(*child) && leafMatches(*child, key, keyLength, depth, maxKeyLength))
    {
      // Leaf found, delete it in inner node
      switch (node->type)
        {
        case NodeType4: eraseNode4(static_cast<Node4*>(node), nodeRef, child); break;
        case NodeType16: eraseNode16(static_cast<Node16*>(node), nodeRef, child); break;
        case NodeType48: eraseNode48(static_cast<Node48*>(node), nodeRef, key[depth]); break;
        case NodeType256: eraseNode256(static_cast<Node256*>(node), nodeRef, key[depth]); break;
        }
    }
  else
    {
      // Recurse
      erase(*child, child, key, keyLength, depth + 1, maxKeyLength);
    }
}

// -----------------------------------------------------------------------------
//! \brief
// -----------------------------------------------------------------------------
//void erase(Node* node, Node** nodeRef, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength)
//{
//  erase(node, &node, key, keyLength, 0u, maxPrefixLength - 1u);
//}

} // namespace art
