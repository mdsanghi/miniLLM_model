name = input("Enter your name: ")
marks = float(input("Enter your marks: "))

if marks >= 40:
    result = "Pass"
else:
    result = "Fail"

print("Hello", name)
print("Your marks are:", marks)
print("Result:", result)