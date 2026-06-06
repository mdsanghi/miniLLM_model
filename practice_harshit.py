# Input two numbers
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Add the numbers
sum = num1 + num2

# Display the result
print("Sum =", sum)
name = input("Enter your name: ")
marks = float(input("Enter your marks: "))

if marks >= 40:
    result = "Pass"
else:
    result = "Fail"

print("Hello", name)
print("Your marks are:", marks)
print("Result:", result)
