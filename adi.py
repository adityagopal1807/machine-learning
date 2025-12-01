import numpy as np
import matplotlib.pyplot as plt

nums = [10, 20, 30]
print(nums[0])  # 10
student={"name":"adi","age":30}
print(student["name"])
print(student["age"])  # 30
def add(a,b):
    return a+b
print(add(6,7.6))
for i in range(6,9):
    print(i)
x=5
while x>=1:
    print(x)
    x-=1    

mat = np.array([[1,2],[3,4]])
mat1=np.array([[5,6],[7,8]])
mat2=np.array([[9,6],[7,8]])
print(mat,mat1)
result = np.dot(np.dot(mat,mat1),mat2)
add=np.add(mat,mat1)
print(result)
print(add)
subs=np.subtract(mat1,mat)
print(subs)
divide=np.divide(mat1,mat)
print(divide)
arr=np.array([2, 4, 6, 8])
sum1=sum(arr)
print(len(arr))
print(sum1)
print(np.mean(arr))
mean1=sum1/len(arr)
print(mean1)
print(np.median(arr))
print(np.var(arr))
plt.plot([1,2,3], [10,20,3])
plt.title("Test Graph")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
print(np.max(arr))
print(np.min(arr))
print(arr * 3)