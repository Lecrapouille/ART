// Adaptive Radix Tree
// Thid code source is a modified version of https://db.in.tum.de/~leis/index/ART.tgz
// made by Viktor Leis, 2012 leis@in.tum.de

#ifndef ART_HPP
#  define ART_HPP

#  include <cstddef>
#  include <cstdint>

namespace art
{

// Constants for the node types
static const int8_t NodeType4 = 0;
static const int8_t NodeType16 = 1;
static const int8_t NodeType48 = 2;
static const int8_t NodeType256 = 3;

// The maximum prefix length for compressed paths stored in the
// header, if the path is longer it is loaded from the database on
// demand
static const uint32_t maxPrefixLength = 9u;

// Shared header of all inner nodes
struct Node
{
  // length of the compressed path (prefix)
  uint32_t prefixLength = 0u;
  // number of non - null children
  uint16_t count = 0u;
  // node type
  int8_t type;
  // compressed path (prefix)
  uint8_t prefix[maxPrefixLength];

  Node(int8_t const type)
    : type(type)
  {}
};

// Node with up to 4 children
struct Node4 : Node
{
  uint8_t key[4] = {};
  Node* child[4] = {};

  Node4()
    : Node(NodeType4)
  {}
};

// Node with up to 16 children
struct Node16 : Node
{
  uint8_t key[16] = {};
  Node* child[16] = {};

  Node16()
    : Node(NodeType16)
  {}
};

static const uint8_t emptyMarker = 48;

// Node with up to 48 children
struct Node48 : Node
{
  uint8_t childIndex[256];
  Node* child[48] = {};

  Node48()
    : Node(NodeType48)
  {
    for (size_t i = 48u; i < 256; ++i)
      childIndex[i] = 0;
  }
};

// Node with up to 256 children
struct Node256 : Node
{
  Node* child[256] = {};

  Node256()
    : Node(NodeType256)
  {}
};

void insert(Node* node, Node** nodeRef, uint8_t key[], uint32_t depth, uintptr_t value, uint32_t maxKeyLength);
void erase(Node* node, Node** nodeRef, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength);
Node* lookup(Node* node, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength);
Node* lookupPessimistic(Node* node, uint8_t key[], uint32_t keyLength, uint32_t depth, uint32_t maxKeyLength);

} // namespace art

#endif // ART_HPP
