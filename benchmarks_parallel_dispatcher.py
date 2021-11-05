import subprocess
import time
import sys
import select


def get_recommended():
    r = subprocess.run(["python3", "benchmarks.py", "list"], stdout=subprocess.PIPE)
    r = r.stdout.decode('utf-8')
    r = r.splitlines()
    return r


pool = []
MAX_POOL_SIZE = 4
already_running = set()
population = 20
while True:

    if len(pool) < MAX_POOL_SIZE:
        for recommended in get_recommended():
            if len(pool) >= MAX_POOL_SIZE:
                break
            name, samples, evaluated_instances = recommended.split()
            code = name + " " + samples
            if code not in already_running and int(evaluated_instances) < population:
                already_running.add(code)
                recommended = ["python3", "benchmarks.py", name, samples]
                p = subprocess.Popen(recommended, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pool.append(p)
                print("Starting: ", recommended)

    for p in pool:
        if p.poll() is not None:
            p.wait()
            already_running.remove(p.args[2]+" "+p.args[3])
            print("Finished", p.args)
    pool = [p for p in pool if p.poll() is None]

    time.sleep(10)
    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        if input() == "exit":
            break
        else:
            print("Unrecognized command")

print("Will exit when all processes finish!")

for p in pool:
    p.wait()
    print("Finished", p.args)
