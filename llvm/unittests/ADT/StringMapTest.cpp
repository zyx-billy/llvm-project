//===- llvm/unittest/ADT/StringMapMap.cpp - StringMap unit tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"
#include <tuple>
using namespace llvm;

namespace {

// Test fixture
class StringMapTest : public testing::Test {
protected:
  StringMap<uint32_t> testMap;

  static const char testKey[];
  static const uint32_t testValue;
  static const char* testKeyFirst;
  static size_t testKeyLength;
  static const std::string testKeyStr;

  void assertEmptyMap() {
    // Size tests
    EXPECT_EQ(0u, testMap.size());
    EXPECT_TRUE(testMap.empty());

    // Iterator tests
    EXPECT_TRUE(testMap.begin() == testMap.end());

    // Lookup tests
    EXPECT_EQ(0u, testMap.count(testKey));
    EXPECT_EQ(0u, testMap.count(StringRef(testKeyFirst, testKeyLength)));
    EXPECT_EQ(0u, testMap.count(testKeyStr));
    EXPECT_TRUE(testMap.find(testKey) == testMap.end());
    EXPECT_TRUE(testMap.find(StringRef(testKeyFirst, testKeyLength)) == 
                testMap.end());
    EXPECT_TRUE(testMap.find(testKeyStr) == testMap.end());
  }

  void assertSingleItemMap() {
    // Size tests
    EXPECT_EQ(1u, testMap.size());
    EXPECT_FALSE(testMap.begin() == testMap.end());
    EXPECT_FALSE(testMap.empty());

    // Iterator tests
    StringMap<uint32_t>::iterator it = testMap.begin();
    EXPECT_STREQ(testKey, it->first().data());
    EXPECT_EQ(testValue, it->second);
    ++it;
    EXPECT_TRUE(it == testMap.end());

    // Lookup tests
    EXPECT_EQ(1u, testMap.count(testKey));
    EXPECT_EQ(1u, testMap.count(StringRef(testKeyFirst, testKeyLength)));
    EXPECT_EQ(1u, testMap.count(testKeyStr));
    EXPECT_TRUE(testMap.find(testKey) == testMap.begin());
    EXPECT_TRUE(testMap.find(StringRef(testKeyFirst, testKeyLength)) == 
                testMap.begin());
    EXPECT_TRUE(testMap.find(testKeyStr) == testMap.begin());
  }
};

const char StringMapTest::testKey[] = "key";
const uint32_t StringMapTest::testValue = 1u;
const char* StringMapTest::testKeyFirst = testKey;
size_t StringMapTest::testKeyLength = sizeof(testKey) - 1;
const std::string StringMapTest::testKeyStr(testKey);

// Empty map tests.
TEST_F(StringMapTest, EmptyMapTest) {
  assertEmptyMap();
}

// Constant map tests.
TEST_F(StringMapTest, ConstEmptyMapTest) {
  const StringMap<uint32_t>& constTestMap = testMap;

  // Size tests
  EXPECT_EQ(0u, constTestMap.size());
  EXPECT_TRUE(constTestMap.empty());

  // Iterator tests
  EXPECT_TRUE(constTestMap.begin() == constTestMap.end());

  // Lookup tests
  EXPECT_EQ(0u, constTestMap.count(testKey));
  EXPECT_EQ(0u, constTestMap.count(StringRef(testKeyFirst, testKeyLength)));
  EXPECT_EQ(0u, constTestMap.count(testKeyStr));
  EXPECT_TRUE(constTestMap.find(testKey) == constTestMap.end());
  EXPECT_TRUE(constTestMap.find(StringRef(testKeyFirst, testKeyLength)) ==
              constTestMap.end());
  EXPECT_TRUE(constTestMap.find(testKeyStr) == constTestMap.end());
}

// A map with a single entry.
TEST_F(StringMapTest, SingleEntryMapTest) {
  testMap[testKey] = testValue;
  assertSingleItemMap();
}

// Test clear() method.
TEST_F(StringMapTest, ClearTest) {
  testMap[testKey] = testValue;
  testMap.clear();
  assertEmptyMap();
}

// Test erase(iterator) method.
TEST_F(StringMapTest, EraseIteratorTest) {
  testMap[testKey] = testValue;
  testMap.erase(testMap.begin());
  assertEmptyMap();
}

// Test erase(value) method.
TEST_F(StringMapTest, EraseValueTest) {
  testMap[testKey] = testValue;
  testMap.erase(testKey);
  assertEmptyMap();
}

// Test inserting two values and erasing one.
TEST_F(StringMapTest, InsertAndEraseTest) {
  testMap[testKey] = testValue;
  testMap["otherKey"] = 2;
  testMap.erase("otherKey");
  assertSingleItemMap();
}

TEST_F(StringMapTest, SmallFullMapTest) {
  // StringMap has a tricky corner case when the map is small (<8 buckets) and
  // it fills up through a balanced pattern of inserts and erases. This can
  // lead to inf-loops in some cases (PR13148) so we test it explicitly here.
  llvm::StringMap<int> Map(2);

  Map["eins"] = 1;
  Map["zwei"] = 2;
  Map["drei"] = 3;
  Map.erase("drei");
  Map.erase("eins");
  Map["veir"] = 4;
  Map["funf"] = 5;

  EXPECT_EQ(3u, Map.size());
  EXPECT_EQ(0, Map.lookup("eins"));
  EXPECT_EQ(2, Map.lookup("zwei"));
  EXPECT_EQ(0, Map.lookup("drei"));
  EXPECT_EQ(4, Map.lookup("veir"));
  EXPECT_EQ(5, Map.lookup("funf"));
}

TEST_F(StringMapTest, CopyCtorTest) {
  llvm::StringMap<int> Map;

  Map["eins"] = 1;
  Map["zwei"] = 2;
  Map["drei"] = 3;
  Map.erase("drei");
  Map.erase("eins");
  Map["veir"] = 4;
  Map["funf"] = 5;

  EXPECT_EQ(3u, Map.size());
  EXPECT_EQ(0, Map.lookup("eins"));
  EXPECT_EQ(2, Map.lookup("zwei"));
  EXPECT_EQ(0, Map.lookup("drei"));
  EXPECT_EQ(4, Map.lookup("veir"));
  EXPECT_EQ(5, Map.lookup("funf"));

  llvm::StringMap<int> Map2(Map);
  EXPECT_EQ(3u, Map2.size());
  EXPECT_EQ(0, Map2.lookup("eins"));
  EXPECT_EQ(2, Map2.lookup("zwei"));
  EXPECT_EQ(0, Map2.lookup("drei"));
  EXPECT_EQ(4, Map2.lookup("veir"));
  EXPECT_EQ(5, Map2.lookup("funf"));
}

// A more complex iteration test.
TEST_F(StringMapTest, IterationTest) {
  bool visited[100];

  // Insert 100 numbers into the map
  for (int i = 0; i < 100; ++i) {
    std::stringstream ss;
    ss << "key_" << i;
    testMap[ss.str()] = i;
    visited[i] = false;
  }

  // Iterate over all numbers and mark each one found.
  for (StringMap<uint32_t>::iterator it = testMap.begin();
      it != testMap.end(); ++it) {
    std::stringstream ss;
    ss << "key_" << it->second;
    ASSERT_STREQ(ss.str().c_str(), it->first().data());
    visited[it->second] = true;
  }

  // Ensure every number was visited.
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(visited[i]) << "Entry #" << i << " was never visited";
  }
}

// Test StringMapEntry::Create() method.
TEST_F(StringMapTest, StringMapEntryTest) {
  StringMap<uint32_t>::value_type* entry =
      StringMap<uint32_t>::value_type::Create(
          StringRef(testKeyFirst, testKeyLength), 1u);
  EXPECT_STREQ(testKey, entry->first().data());
  EXPECT_EQ(1u, entry->second);
  free(entry);
}

// Test insert() method.
TEST_F(StringMapTest, InsertTest) {
  SCOPED_TRACE("InsertTest");
  testMap.insert(
      StringMap<uint32_t>::value_type::Create(
          StringRef(testKeyFirst, testKeyLength),
          testMap.getAllocator(), 1u));
  assertSingleItemMap();
}

// Test insert(pair<K, V>) method
TEST_F(StringMapTest, InsertPairTest) {
  bool Inserted;
  StringMap<uint32_t>::iterator NewIt;
  std::tie(NewIt, Inserted) =
      testMap.insert(std::make_pair(testKeyFirst, testValue));
  EXPECT_EQ(1u, testMap.size());
  EXPECT_EQ(testValue, testMap[testKeyFirst]);
  EXPECT_EQ(testKeyFirst, NewIt->first());
  EXPECT_EQ(testValue, NewIt->second);
  EXPECT_TRUE(Inserted);

  StringMap<uint32_t>::iterator ExistingIt;
  std::tie(ExistingIt, Inserted) =
      testMap.insert(std::make_pair(testKeyFirst, testValue + 1));
  EXPECT_EQ(1u, testMap.size());
  EXPECT_EQ(testValue, testMap[testKeyFirst]);
  EXPECT_FALSE(Inserted);
  EXPECT_EQ(NewIt, ExistingIt);
}

// Test insert(pair<K, V>) method when rehashing occurs
TEST_F(StringMapTest, InsertRehashingPairTest) {
  // Check that the correct iterator is returned when the inserted element is
  // moved to a different bucket during internal rehashing. This depends on
  // the particular key, and the implementation of StringMap and HashString.
  // Changes to those might result in this test not actually checking that.
  StringMap<uint32_t> t(0);
  EXPECT_EQ(0u, t.getNumBuckets());

  StringMap<uint32_t>::iterator It =
    t.insert(std::make_pair("abcdef", 42)).first;
  EXPECT_EQ(16u, t.getNumBuckets());
  EXPECT_EQ("abcdef", It->first());
  EXPECT_EQ(42u, It->second);
}

TEST_F(StringMapTest, IterMapKeys) {
  StringMap<int> Map;
  Map["A"] = 1;
  Map["B"] = 2;
  Map["C"] = 3;
  Map["D"] = 3;

  auto Keys = to_vector<4>(Map.keys());
  std::sort(Keys.begin(), Keys.end());

  SmallVector<StringRef, 4> Expected = {"A", "B", "C", "D"};
  EXPECT_EQ(Expected, Keys);
}

TEST_F(StringMapTest, IterSetKeys) {
  StringSet<> Set;
  Set.insert("A");
  Set.insert("B");
  Set.insert("C");
  Set.insert("D");

  auto Keys = to_vector<4>(Set.keys());
  std::sort(Keys.begin(), Keys.end());

  SmallVector<StringRef, 4> Expected = {"A", "B", "C", "D"};
  EXPECT_EQ(Expected, Keys);
}

// Create a non-default constructable value
struct StringMapTestStruct {
  StringMapTestStruct(int i) : i(i) {}
  StringMapTestStruct() = delete;
  int i;
};

TEST_F(StringMapTest, NonDefaultConstructable) {
  StringMap<StringMapTestStruct> t;
  t.insert(std::make_pair("Test", StringMapTestStruct(123)));
  StringMap<StringMapTestStruct>::iterator iter = t.find("Test");
  ASSERT_NE(iter, t.end());
  ASSERT_EQ(iter->second.i, 123);
}

struct Immovable {
  Immovable() {}
  Immovable(Immovable&&) = delete; // will disable the other special members
};

struct MoveOnly {
  int i;
  MoveOnly(int i) : i(i) {}
  MoveOnly(const Immovable&) : i(0) {}
  MoveOnly(MoveOnly &&RHS) : i(RHS.i) {}
  MoveOnly &operator=(MoveOnly &&RHS) {
    i = RHS.i;
    return *this;
  }

private:
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;
};

TEST_F(StringMapTest, MoveOnly) {
  StringMap<MoveOnly> t;
  t.insert(std::make_pair("Test", MoveOnly(42)));
  StringRef Key = "Test";
  StringMapEntry<MoveOnly>::Create(Key, MoveOnly(42))
      ->Destroy();
}

TEST_F(StringMapTest, CtorArg) {
  StringRef Key = "Test";
  StringMapEntry<MoveOnly>::Create(Key, Immovable())
      ->Destroy();
}

TEST_F(StringMapTest, MoveConstruct) {
  StringMap<int> A;
  A["x"] = 42;
  StringMap<int> B = std::move(A);
  ASSERT_EQ(A.size(), 0u);
  ASSERT_EQ(B.size(), 1u);
  ASSERT_EQ(B["x"], 42);
  ASSERT_EQ(B.count("y"), 0u);
}

TEST_F(StringMapTest, MoveAssignment) {
  StringMap<int> A;
  A["x"] = 42;
  StringMap<int> B;
  B["y"] = 117;
  A = std::move(B);
  ASSERT_EQ(A.size(), 1u);
  ASSERT_EQ(B.size(), 0u);
  ASSERT_EQ(A["y"], 117);
  ASSERT_EQ(B.count("x"), 0u);
}

struct Countable {
  int &InstanceCount;
  int Number;
  Countable(int Number, int &InstanceCount)
      : InstanceCount(InstanceCount), Number(Number) {
    ++InstanceCount;
  }
  Countable(Countable &&C) : InstanceCount(C.InstanceCount), Number(C.Number) {
    ++InstanceCount;
    C.Number = -1;
  }
  Countable(const Countable &C)
      : InstanceCount(C.InstanceCount), Number(C.Number) {
    ++InstanceCount;
  }
  Countable &operator=(Countable C) {
    Number = C.Number;
    return *this;
  }
  ~Countable() { --InstanceCount; }
};

TEST_F(StringMapTest, MoveDtor) {
  int InstanceCount = 0;
  StringMap<Countable> A;
  A.insert(std::make_pair("x", Countable(42, InstanceCount)));
  ASSERT_EQ(InstanceCount, 1);
  auto I = A.find("x");
  ASSERT_NE(I, A.end());
  ASSERT_EQ(I->second.Number, 42);

  StringMap<Countable> B;
  B = std::move(A);
  ASSERT_EQ(InstanceCount, 1);
  ASSERT_TRUE(A.empty());
  I = B.find("x");
  ASSERT_NE(I, B.end());
  ASSERT_EQ(I->second.Number, 42);

  B = StringMap<Countable>();
  ASSERT_EQ(InstanceCount, 0);
  ASSERT_TRUE(B.empty());
}

namespace {
// Simple class that counts how many moves and copy happens when growing a map
struct CountCtorCopyAndMove {
  static unsigned Ctor;
  static unsigned Move;
  static unsigned Copy;
  int Data = 0;
  CountCtorCopyAndMove(int Data) : Data(Data) { Ctor++; }
  CountCtorCopyAndMove() { Ctor++; }

  CountCtorCopyAndMove(const CountCtorCopyAndMove &) { Copy++; }
  CountCtorCopyAndMove &operator=(const CountCtorCopyAndMove &) {
    Copy++;
    return *this;
  }
  CountCtorCopyAndMove(CountCtorCopyAndMove &&) { Move++; }
  CountCtorCopyAndMove &operator=(const CountCtorCopyAndMove &&) {
    Move++;
    return *this;
  }
};
unsigned CountCtorCopyAndMove::Copy = 0;
unsigned CountCtorCopyAndMove::Move = 0;
unsigned CountCtorCopyAndMove::Ctor = 0;

} // anonymous namespace

// Make sure creating the map with an initial size of N actually gives us enough
// buckets to insert N items without increasing allocation size.
TEST(StringMapCustomTest, InitialSizeTest) {
  // 1 is an "edge value", 32 is an arbitrary power of two, and 67 is an
  // arbitrary prime, picked without any good reason.
  for (auto Size : {1, 32, 67}) {
    StringMap<CountCtorCopyAndMove> Map(Size);
    auto NumBuckets = Map.getNumBuckets();
    CountCtorCopyAndMove::Move = 0;
    CountCtorCopyAndMove::Copy = 0;
    for (int i = 0; i < Size; ++i)
      Map.insert(std::pair<std::string, CountCtorCopyAndMove>(
          std::piecewise_construct, std::forward_as_tuple(Twine(i).str()),
          std::forward_as_tuple(i)));
    // After the inital move, the map will move the Elts in the Entry.
    EXPECT_EQ((unsigned)Size * 2, CountCtorCopyAndMove::Move);
    // We copy once the pair from the Elts vector
    EXPECT_EQ(0u, CountCtorCopyAndMove::Copy);
    // Check that the map didn't grow
    EXPECT_EQ(Map.getNumBuckets(), NumBuckets);
  }
}

TEST(StringMapCustomTest, BracketOperatorCtor) {
  StringMap<CountCtorCopyAndMove> Map;
  CountCtorCopyAndMove::Ctor = 0;
  Map["abcd"];
  EXPECT_EQ(1u, CountCtorCopyAndMove::Ctor);
  // Test that operator[] does not create a value when it is already in the map
  CountCtorCopyAndMove::Ctor = 0;
  Map["abcd"];
  EXPECT_EQ(0u, CountCtorCopyAndMove::Ctor);
}

namespace {
struct NonMoveableNonCopyableType {
  int Data = 0;
  NonMoveableNonCopyableType() = default;
  NonMoveableNonCopyableType(int Data) : Data(Data) {}
  NonMoveableNonCopyableType(const NonMoveableNonCopyableType &) = delete;
  NonMoveableNonCopyableType(NonMoveableNonCopyableType &&) = delete;
};
}

// Test that we can "emplace" an element in the map without involving map/move
TEST(StringMapCustomTest, EmplaceTest) {
  StringMap<NonMoveableNonCopyableType> Map;
  Map.try_emplace("abcd", 42);
  EXPECT_EQ(1u, Map.count("abcd"));
  EXPECT_EQ(42, Map["abcd"].Data);
}

} // end anonymous namespace
