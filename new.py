file=open("hello.txt","w")
file.write("Hello Deepika")
file.close()
file=open("hello.txt","a")
file.write("\nWelcome to Python \n LookUp")
file.close()
with open("hello.txt","r") as f:
    print(f.read())
                    