import subprocess, sys

ADB = r"C:\Users\Administrator\platform-tools\adb.exe"

def run_adb(*args):
    try:
        r = subprocess.run([ADB, *args], capture_output=True, timeout=15, text=True)
        return r.stdout
    except Exception as e:
        return str(e)

out = run_adb("shell", "dumpsys", "usagestats")
with open("usagestats_dump.txt", "w", encoding="utf-8") as f:
    f.write(out)

print(f"Dumped {len(out)} characters into usagestats_dump.txt")
print("First 20 lines:")
lines = out.splitlines()
for l in lines[:20]:
    print(l)

out2 = run_adb("shell", "dumpsys", "usagestats", "app")
print("\nFirst 20 lines of usagestats app:")
lines = out2.splitlines()
for l in lines[:20]:
    print(l)
