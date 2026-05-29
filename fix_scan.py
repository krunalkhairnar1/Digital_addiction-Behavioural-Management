
import sys, io, subprocess, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ADB = r"C:\Users\Administrator\platform-tools\adb.exe"

def adb(*args, timeout=40):
    r = subprocess.run([ADB, *args], capture_output=True, timeout=timeout)
    return r.stdout.decode('utf-8', errors='replace')

# Build UID map
uid_map = {}
out = adb("shell", "pm", "list", "packages", "-U")
for line in out.splitlines():
    m = re.match(r"package:([\w.]+)\s+uid:(\d+)", line.strip())
    if m:
        pkg, uid = m.group(1), int(m.group(2))
        uid_map[uid] = pkg

print(f"Total UIDs mapped: {len(uid_map)}")

# Get batterystats and find all u0aXXX entries
batt = adb("shell", "dumpsys", "batterystats", "--charged", timeout=40)
lines = batt.splitlines()
print(f"Total batterystats lines: {len(lines)}")

# Find all UID lines in summary section
uid_re = re.compile(r"UID u0a(\d+):")
screen_re = re.compile(r"screen=[\d.]+ \(([\dhms ]+?)\)")
fg_re = re.compile(r"cpu:fg=[\d.]+ \(([\dhms ]+?)\)")

def hms_to_min(s):
    total = 0
    h = re.search(r"(\d+)h", s); total += int(h.group(1))*60 if h else 0
    m = re.search(r"(\d+)m", s); total += int(m.group(1)) if m else 0
    s2 = re.search(r"(\d+)s", s); total += int(s2.group(1))/60 if s2 else 0
    return round(total, 1)

print("\n=== All UID entries in batterystats ===")
found_uids = []
for line in lines:
    m = uid_re.search(line)
    if m:
        suffix = int(m.group(1))
        actual_uid = 10000 + suffix
        pkg = uid_map.get(actual_uid, "UNKNOWN")
        
        sm = screen_re.search(line)
        fm = fg_re.search(line)
        screen_min = hms_to_min(sm.group(1)) if sm else 0
        fg_min = hms_to_min(fm.group(1)) if fm else 0
        
        found_uids.append((actual_uid, suffix, pkg, screen_min, fg_min, line.strip()[:80]))

# Sort by screen time
found_uids.sort(key=lambda x: x[3], reverse=True)
for uid, suffix, pkg, screen_min, fg_min, raw in found_uids[:30]:
    print(f"  u0a{suffix:3d} (uid={uid}) pkg={pkg:40s} screen={screen_min:6.1f}min fg={fg_min:6.1f}min")
