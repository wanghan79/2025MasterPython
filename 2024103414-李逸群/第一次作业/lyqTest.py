import copy
#Test1:int is a stable type
a = 1
print("Oringin 'int' variable 'a' address"+str(id(a)))
a = 2
print("Re-assigned 'int' vvariable 'a' address"+str(id(a)))
b = a
print("Another 'int' variable 'b' shared 'a' address: "+str(id(b)))
# Test2: str is a stable type
a = "Python is great!"
print(id(a))
a = "Python is so great!"
print(id(a))
b = a
print(id(a))
print(id(b))


# Test3: tuple is a stable type
a = (1,2,3)
b = a
print(id(a))
a = (5,6,7)
print(id(a))
c = a + b
print(id(c))
c = c + a
print(id(c))


a = [1,2,3]
b = a
print(id(a))
a = [5,6,7]
print(id(a))
a[0] = 10
print(id(a))
a.append(9)
print(id(a))


# Test4: set is a flexible type
a = {1, 2, 3}
print(id(a))
a.add(4)
print(id(a))
b = a
print(id(a))
print(id(b))


# Test5: dict is a flexible type
a = {1:"one", 2:"two", 3:"three"}
print(id(a))
a.update({4:"four"})
print(id(a))


b = a.copy()
print(id(a))
print(id(b))

b.clear()
print(a)
print(b)


# Test6: tuple,list,set,dict are dropbox of mixed data types
a = (1,"str",[5,6,7])
print(a)

a = [1,"str",[5,6,7]]
print(a)
print(id(a))
a.append({"aaa":100})
print(a)
print(id(a))

strtmp = "tmp string"
a.append(strtmp)
print(a)

print(id(a))
a[1] = "another str"
print(id(a))

print(id(a))
a = [1, "str", [5, 6, 7], {"key": "value"}]
a[3].update({"new_key": "another_str"})
print(a)



# Test7: copy() and deepcopy()
lista = [4,5]
listb = [1,2,3,lista]
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



# 预设 3 名学生的信息
# 每个学生的选课信息使用字典存储
student1 = ("001", "Alice", "Female", "alice.com", "1234567890", {"Math": 85, "Physics": 90})
student2 = ("002", "Bob", "Male", "bob.com", "0987654321", {"Chemistry": 78, "Biology": 88})
student3 = ("003", "Charlie", "Male", "charlie.com", "1122334455", {"English": 92, "History": 86})

# 使用 tuple 存储所有学生信息
students_tuple = (student1, student2, student3)

# 使用 list 存储所有学生信息
students_list = [list(student1), list(student2), list(student3)]

# 使用 dict 存储所有学生信息，学号作为键
students_dict = {
    student1
[0]: {
        "姓名": student1[1],
        "性别": student1[2],
        "邮箱": student1[3],
        "电话": student1[4],
        "选课信息": student1[5]
    },
    student2
[0]: {
        "姓名": student2[1],
        "性别": student2[2],
        "邮箱": student2[3],
        "电话": student2[4],
        "选课信息": student2[5]
    },
    student3
[0]: {
        "姓名": student3[1],
        "性别": student3[2],
        "邮箱": student3[3],
        "电话": student3[4],
        "选课信息": student3[5]
    }
}

# 打印不同数据结构存储的学生信息
print("使用 tuple 存储的学生信息：")
print(students_tuple)
print("\n使用 list 存储的学生信息：")
print(students_list)
print("\n使用 dict 存储的学生信息：")
print(students_dict)


