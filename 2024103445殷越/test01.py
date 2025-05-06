import copy

# Test 1: int is a stable type
a = 1
print("Original 'int' variable 'a' address: " + str(id(a)))
a = 2
print("Re-assigned 'int' variable 'a' address: " + str(id(a)))
b = 2
print("Another 'int' variable 'b' shared 'a' address: " + str(id(b)))

# Test 2: str is a stable type
a = "Python is great!"
print(id(a))
a = "Python is so great!"
print(id(a))
b = a
print(id(a))
print(id(b))

# Test 3: tuple is a stable type
a = (1, 2, 3)
b = a
print(id(a))
a = (5, 6, 7)
print(id(a))
c = a
print(id(c))
c = c + (8,)
print(id(c))

## Test 4: list is a flexible type
a = [1, 2, 3]
b = a
print(id(a))
a = [5, 6, 7]
print(id(a))
a[0] = 10
print(id(a))
a.append(9)
print(id(a))

# Test 4: set is a flexible type
a = {1, 2, 3}
print(id(a))
a.add(4)
print(id(a))
b = a
print(id(b))

# Test 5: dict is a flexible type
a = {1: "one", 2: "two", 3: "three"}
print(id(a))
a.update({4: "four"})
print(id(a))

# Copying and clearing dictionaries
b = a
print(id(a))
print(id(b))
# b.clear()
# print(a)
# print(b)

b = a.copy()
print(id(a))
print(id(b))
print(b)
b.clear()
print(b)

# Test 6: tuple, list, set, dict are mixed data type containers
a = (1, "str", [5, 6, 7])
print(a)
print(id(a))
a = [1, "str", [5, 6, 7]]
print(a)
print(id(a))
a.append({"aaa", 000})
print(a)
print(id(a))

str_tmp = "tmp string"
a.append(str_tmp)
print(a)
print(id(a))

a[1] = "another str"
print(id(a))

a[3].add("another str")
print(id(a))

# Test 8: copy() and deepcopy()
lista = [4, 5]
listb = [1, 2, 3, lista]
lista.append(6)
listc = copy.copy(listb)
listd = copy.deepcopy(listb)

print(id(listb))
print(id(listc))
print(id(listd))

lista.append(7)
print(id(listb))
print(id(listc))
print(id(listd))

# Practice Exercises:
# 1. Use tuple to store student info (ID, name, gender, email, phone, selected courses as dictionary)
# 2. Use list to store the above tuples.
# 3. Use dict to store the above information.
