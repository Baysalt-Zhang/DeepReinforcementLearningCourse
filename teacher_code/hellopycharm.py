def changeme(mylist):
    mylist.append([1,2,3,4])
    print("in:", mylist)
    return

mylist = [10,20,30]
changeme(mylist)
print("out:",mylist)