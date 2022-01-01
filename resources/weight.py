import subprocess
# invoke process
process = subprocess.Popen(["python", "/home/jetson/main/example.py"], shell=False, stdout=subprocess.PIPE,stderr=subprocess.PIPE)

# Poll process.stdout to show stdout live

weight=0.0
prevWeight=0.0
while True:
    output = process.stdout.readline()
    if process.poll() is not None:
        break
    if output:
        prevWeight=weight
        weight=float(output.decode())
        print(weight)
        if abs(prevWeight-weight)>50:
            print("Detected.")
        